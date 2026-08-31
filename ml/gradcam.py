"""
Robust Grad-CAM implementation for NAV-SMFS.

Supports:
- EfficientNet-B0
- PyTorch models
- Binary single-logit output
- Two-class output
- Real / AI Generated classification
- Face-level Grad-CAM
- Blue -> Green -> Yellow -> Red heatmap
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from pathlib import Path


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer=None):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None
        self.backward_handle = None

        # ----------------------------------------------------
        # Find target layer
        # ----------------------------------------------------

        if self.target_layer is None:
            self.target_layer = self._find_target_layer()

        print(
            f"✅ Grad-CAM target layer: "
            f"{self._get_layer_name(self.target_layer)}"
        )

        # ----------------------------------------------------
        # Register hooks
        # ----------------------------------------------------

        self._register_hooks()

    # ========================================================
    # GET LAYER NAME
    # ========================================================

    def _get_layer_name(self, target):

        for name, module in self.model.named_modules():

            if module is target:
                return name

        return str(target)

    # ========================================================
    # FIND TARGET LAYER
    # ========================================================

    def _find_target_layer(self):

        print("\n🔍 Searching for best Grad-CAM target layer...")

        # ----------------------------------------------------
        # First preference:
        # Last Conv2d in the model
        # ----------------------------------------------------

        last_conv = None
        last_conv_name = None

        for name, module in self.model.named_modules():

            if isinstance(module, torch.nn.Conv2d):

                last_conv = module
                last_conv_name = name

        if last_conv is not None:

            print(
                f"✅ Last Conv2d found: "
                f"{last_conv_name}"
            )

            return last_conv

        # ----------------------------------------------------
        # Fallback:
        # Conv-like layer
        # ----------------------------------------------------

        possible_layers = []

        for name, module in self.model.named_modules():

            if (
                isinstance(module, torch.nn.BatchNorm2d)
                or isinstance(module, torch.nn.ReLU)
            ):

                possible_layers.append(
                    (name, module)
                )

        if possible_layers:

            name, module = possible_layers[-1]

            print(
                f"⚠️ No Conv2d found."
                f"Using fallback: {name}"
            )

            return module

        raise RuntimeError(
            "❌ Could not find suitable Grad-CAM target layer."
        )

    # ========================================================
    # REGISTER HOOKS
    # ========================================================

    def _register_hooks(self):

        # ----------------------------------------------------
        # Forward hook
        # ----------------------------------------------------

        def forward_hook(
            module,
            input_data,
            output
        ):

            self.activations = output

            print(
                "   🎯 Forward hook:"
                f" {tuple(output.shape)}"
            )

        # ----------------------------------------------------
        # Full backward hook
        # ----------------------------------------------------

        def backward_hook(
            module,
            grad_input,
            grad_output
        ):

            if grad_output is not None:

                self.gradients = grad_output[0]

                if self.gradients is not None:

                    print(
                        "   🎯 Backward hook:"
                        f" {tuple(self.gradients.shape)}"
                    )

        # IMPORTANT:
        # register_full_backward_hook instead of deprecated
        # register_backward_hook

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                backward_hook
            )
        )

    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

            self.forward_handle = None

        if self.backward_handle is not None:

            self.backward_handle.remove()

            self.backward_handle = None

    # ========================================================
    # GENERATE CAM
    # ========================================================

    def generate_cam(
        self,
        input_tensor,
        target_class=None
    ):

        print("\n🔥 Generating Grad-CAM...")

        self.model.eval()

        # Clear previous data

        self.activations = None
        self.gradients = None

        # Clear model gradients

        self.model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        print("1️⃣ Forward pass...")

        output = self.model(
            input_tensor
        )

        print(
            f"   Output shape: "
            f"{tuple(output.shape)}"
        )

        print(
            f"   Output values: "
            f"{output.detach().cpu().numpy()}"
        )

        # ====================================================
        # HANDLE OUTPUT FORMAT
        # ====================================================

        # ----------------------------------------------------
        # CASE 1:
        # Two-class output
        #
        # shape = [1, 2]
        # class 0 = fake
        # class 1 = real
        # ----------------------------------------------------

        if output.ndim == 2 and output.shape[1] == 2:

            print(
                "   📌 Detected TWO-CLASS output"
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[
                    0,
                    prediction
                ].item()
                * 100
            )

            if target_class is None:

                target_class = prediction

            target_score = output[
                0,
                target_class
            ]

            print(
                f"   Prediction: {prediction}"
            )

            print(
                f"   Target class: {target_class}"
            )

            print(
                f"   Confidence: "
                f"{confidence:.2f}%"
            )

        # ----------------------------------------------------
        # CASE 2:
        # Single binary logit
        #
        # shape = [1]
        # or [1,1]
        # ----------------------------------------------------

        else:

            print(
                "   📌 Detected SINGLE-LOGIT binary output"
            )

            logit = output.reshape(-1)[0]

            probability_real = torch.sigmoid(
                logit
            )

            probability_fake = (
                1.0 - probability_real
            )

            if probability_real >= 0.5:

                prediction = 1

                confidence = (
                    probability_real.item()
                    * 100
                )

            else:

                prediction = 0

                confidence = (
                    probability_fake.item()
                    * 100
                )

            if target_class is None:

                target_class = prediction

            print(
                f"   Real probability: "
                f"{probability_real.item() * 100:.2f}%"
            )

            print(
                f"   Fake probability: "
                f"{probability_fake.item() * 100:.2f}%"
            )

            print(
                f"   Prediction: "
                f"{prediction}"
            )

            print(
                f"   Target class: "
                f"{target_class}"
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # If class 1 = Real:
            #
            # Real target -> +logit
            # Fake target -> -logit
            # ------------------------------------------------

            if target_class == 1:

                target_score = logit

            else:

                target_score = -logit

        # ====================================================
        # BACKWARD
        # ====================================================

        print(
            "2️⃣ Computing gradients..."
        )

        target_score.backward(
            retain_graph=False
        )

        # ====================================================
        # CHECK ACTIVATIONS
        # ====================================================

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations are None. "
                "Target layer hook did not execute."
            )

        # ====================================================
        # CHECK GRADIENTS
        # ====================================================

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients are None. "
                "Backward hook did not receive gradients."
            )

        activations = self.activations
        gradients = self.gradients

        print(
            f"3️⃣ Activations: "
            f"{tuple(activations.shape)}"
        )

        print(
            f"4️⃣ Gradients: "
            f"{tuple(gradients.shape)}"
        )

        # ====================================================
        # ENSURE 4D
        # ====================================================

        if activations.ndim != 4:

            raise RuntimeError(
                "Expected 4D activation tensor "
                f"(B,C,H,W), got "
                f"{tuple(activations.shape)}"
            )

        if gradients.ndim != 4:

            raise RuntimeError(
                "Expected 4D gradient tensor "
                f"(B,C,H,W), got "
                f"{tuple(gradients.shape)}"
            )

        # ====================================================
        # GLOBAL AVERAGE POOLING
        # ====================================================

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        print(
            f"5️⃣ Gradient weights: "
            f"{tuple(weights.shape)}"
        )

        # ====================================================
        # WEIGHT ACTIVATIONS
        # ====================================================

        weighted_activations = (
            weights * activations
        )

        # ====================================================
        # SUM CHANNELS
        # ====================================================

        cam = torch.sum(
            weighted_activations,
            dim=1
        )

        # ====================================================
        # RELU
        # ====================================================

        cam = F.relu(
            cam
        )

        # ====================================================
        # REMOVE BATCH DIMENSION
        # ====================================================

        cam = cam[0]

        # ====================================================
        # CPU NUMPY
        # ====================================================

        cam = (
            cam.detach()
            .cpu()
            .numpy()
        )

        print(
            f"6️⃣ Raw CAM shape: "
            f"{cam.shape}"
        )

        print(
            f"   Raw CAM min: "
            f"{cam.min():.6f}"
        )

        print(
            f"   Raw CAM max: "
            f"{cam.max():.6f}"
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        cam_min = cam.min()
        cam_max = cam.max()

        if (
            not np.isfinite(cam_min)
            or
            not np.isfinite(cam_max)
        ):

            raise RuntimeError(
                "CAM contains NaN or infinite values."
            )

        if (
            cam_max - cam_min
            < 1e-8
        ):

            print(
                "⚠️ CAM has almost no variation."
            )

            # Return a small zero map instead of fake
            # artificial activation.

            cam = np.zeros_like(
                cam,
                dtype=np.float32
            )

        else:

            cam = (
                cam - cam_min
            ) / (
                cam_max
                - cam_min
                + 1e-8
            )

        cam = np.clip(
            cam,
            0.0,
            1.0
        )

        print(
            f"7️⃣ Normalized CAM range: "
            f"[{cam.min():.4f}, "
            f"{cam.max():.4f}]"
        )

        return (
            cam,
            prediction,
            confidence
        )

    # ========================================================
    # HEATMAP OVERLAY
    # ========================================================

    def generate_heatmap_overlay(
        self,
        image_path,
        cam,
        alpha=0.50
    ):

        print(
            "\n🎨 Creating Grad-CAM heatmap overlay..."
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            raise RuntimeError(
                f"Could not load image: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        original = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        height, width = (
            original.shape[:2]
        )

        print(
            f"   Original image: "
            f"{width} x {height}"
        )

        # ----------------------------------------------------
        # Resize CAM
        # ----------------------------------------------------

        cam_resized = cv2.resize(
            cam,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Normalize again
        # ----------------------------------------------------

        cam_resized = np.clip(
            cam_resized,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Convert to uint8
        # ----------------------------------------------------

        cam_uint8 = np.uint8(
            cam_resized * 255
        )

        # ----------------------------------------------------
        # JET COLOR MAP
        #
        # OpenCV JET:
        #
        # Blue  = low
        # Green = medium
        # Yellow = high
        # Red = very high
        # ----------------------------------------------------

        heatmap = cv2.applyColorMap(
            cam_uint8,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            1.0 - alpha,
            heatmap,
            alpha,
            0
        )

        print(
            f"   Overlay shape: "
            f"{overlay.shape}"
        )

        return overlay


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_gradcam_for_face(
    model,
    face_image_path,
    device,
    target_class=None
):

    print("\n" + "=" * 70)
    print("🔥 GRAD-CAM GENERATION STARTED")
    print("=" * 70)

    print(
        f"📁 Face image: "
        f"{face_image_path}"
    )

    print(
        f"🖥️ Device: "
        f"{device}"
    )

    print(
        f"🎯 Target class: "
        f"{target_class}"
    )

    gradcam = None

    try:

        # ====================================================
        # CHECK FILE
        # ====================================================

        if not Path(
            face_image_path
        ).exists():

            raise FileNotFoundError(
                f"Face image not found: "
                f"{face_image_path}"
            )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(
            face_image_path
        ).convert("RGB")

        print(
            f"📷 Original face size: "
            f"{image.size}"
        )

        # ====================================================
        # PREPROCESS
        #
        # SAME AS MODEL LOADER
        # ====================================================

        from torchvision import transforms

        transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        input_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )

        print(
            f"🧠 Input tensor: "
            f"{tuple(input_tensor.shape)}"
        )

        print(
            f"   Requires grad: "
            f"{input_tensor.requires_grad}"
        )

        # ====================================================
        # MODEL
        # ====================================================

        model = model.to(device)
        model.eval()

        # ====================================================
        # INITIALIZE GRAD-CAM
        # ====================================================

        print(
            "\n🔧 Initializing Grad-CAM..."
        )

        gradcam = GradCAM(
            model
        )

        print(
            f"🎯 Using target layer: "
            f"{gradcam._get_layer_name(gradcam.target_layer)}"
        )

        # ====================================================
        # GENERATE CAM
        # ====================================================

        cam, prediction, confidence = (
            gradcam.generate_cam(
                input_tensor,
                target_class
            )
        )

        print(
            f"📊 Prediction: "
            f"{prediction}"
        )

        print(
            f"📊 Confidence: "
            f"{confidence:.2f}%"
        )

        # ====================================================
        # GENERATE OVERLAY
        # ====================================================

        overlay = (
            gradcam.generate_heatmap_overlay(
                face_image_path,
                cam,
                alpha=0.50
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "✅ GRAD-CAM GENERATION SUCCESSFUL"
        )

        print(
            "=" * 70
        )

        return {

            "cam": cam,

            "prediction": prediction,

            "confidence": confidence,

            "overlay": overlay,

            "target_class": target_class,

            "success": True

        }

    except Exception as e:

        print(
            "\n" + "=" * 70
        )

        print(
            "❌ GRAD-CAM GENERATION FAILED"
        )

        print(
            f"❌ {type(e).__name__}: {e}"
        )

        print(
            "=" * 70
        )

        import traceback

        traceback.print_exc()

        return {

            "cam": None,

            "prediction": None,

            "confidence": 0.0,

            "overlay": None,

            "target_class": target_class,

            "success": False,

            "error": str(e)

        }

    finally:

        # ====================================================
        # CLEAN HOOKS
        # ====================================================

        if gradcam is not None:

            gradcam.remove_hooks()

            print(
                "🧹 Grad-CAM hooks removed."
            )