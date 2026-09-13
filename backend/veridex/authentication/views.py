import os
import sys
from pathlib import Path

import cv2
import numpy as np

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse, request
from django.http import FileResponse
from django.contrib.auth.models import User

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ML MODEL
# ============================================================

try:
    from ml.model_loader import predict_image

    print("ML module imported successfully!")

except ImportError as e:

    print(f"ML module import error: {e}")

    def predict_image(image_path):
        return {
            "prediction": "AI Generated",
            "confidence": 0.0,
            "real_probability": 0.0,
            "fake_probability": 0.0,
            "error": str(e)
        }


# ============================================================
# DATABASE MODEL
# ============================================================

from authentication.models import MediaUpload


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

from authentication.utils import (
    detect_faces,
    detect_faces_video,
    draw_faces_on_image,
    is_image_file,
    is_video_file,
    get_file_extension
)


# ============================================================
# MTCNN FACE DETECTOR
# ============================================================

try:

    from mtcnn import MTCNN

    detector = MTCNN()

    print("MTCNN detector initialized successfully!")

except ImportError as e:

    print(f"MTCNN import error: {e}")

    detector = None

    
from django.shortcuts import render
def react_frontend(request):
    return render(request, "react/index.html")
# ============================================================
# STATIC HEATMAP GENERATOR
# ============================================================

def generate_static_heatmap(
    face_crop,
    output_path,
    prediction="AI Generated"
):
    """
    Generate a static heatmap-style visualization.

    NOTE:
    This is a visualization fallback and is NOT model-derived
    Grad-CAM. It creates a smooth attention-style heatmap over
    the detected face region.

    Color interpretation:

    Blue   -> Low significance
    Green  -> Moderate significance
    Yellow -> Significant contribution
    Red    -> Highly significant contribution
    """

    try:

        # ====================================================
        # VALIDATE FACE CROP
        # ====================================================

        if face_crop is None:

            print(
                "❌ Static heatmap: face crop is None"
            )

            return None

        if face_crop.size == 0:

            print(
                "❌ Static heatmap: face crop is empty"
            )

            return None

        # ====================================================
        # GET SIZE
        # ====================================================

        height, width = face_crop.shape[:2]

        if height < 20 or width < 20:

            print(
                "❌ Static heatmap: face crop too small"
            )

            return None

        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )

        # ====================================================
        # LOCAL CONTRAST
        # ====================================================

        blurred = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=15
        )

        contrast = cv2.absdiff(
            gray,
            blurred
        )

        contrast = cv2.normalize(
            contrast,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        contrast = contrast.astype(
            np.float32
        )

        # ====================================================
        # EDGE INFORMATION
        # ====================================================

        edges = cv2.Laplacian(
            gray,
            cv2.CV_32F
        )

        edges = np.abs(edges)

        edges = cv2.GaussianBlur(
            edges,
            (0, 0),
            sigmaX=7
        )

        if edges.max() > 0:

            edges = (
                edges /
                edges.max()
            ) * 255

        # ====================================================
        # CENTER PRIOR
        # ====================================================

        yy, xx = np.mgrid[
            0:height,
            0:width
        ]

        cx = width / 2.0
        cy = height / 2.0

        sigma_x = width * 0.42
        sigma_y = height * 0.48

        center_map = np.exp(
            -(
                (
                    (xx - cx) ** 2
                    /
                    (2 * sigma_x ** 2)
                )
                +
                (
                    (yy - cy) ** 2
                    /
                    (2 * sigma_y ** 2)
                )
            )
        )

        center_map = (
            center_map * 255
        ).astype(
            np.float32
        )

        # ====================================================
        # COMBINE FEATURES
        # ====================================================

        heat = (
            0.55 * center_map
            +
            0.25 * contrast
            +
            0.20 * edges
        )

        # ====================================================
        # SMOOTH HEATMAP
        # ====================================================

        heat = cv2.GaussianBlur(
            heat,
            (0, 0),
            sigmaX=9
        )

        # ====================================================
        # NORMALIZE
        # ====================================================

        heat = cv2.normalize(
            heat,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        heat = np.clip(
            heat,
            0,
            255
        ).astype(
            np.uint8
        )

        # ====================================================
        # APPLY JET COLOR MAP
        #
        # Blue -> Green -> Yellow -> Red
        # ====================================================

        heatmap_bgr = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )

        # ====================================================
        # ORIGINAL FACE
        # ====================================================

        original_bgr = face_crop.copy()

        # ====================================================
        # OVERLAY
        # ====================================================

        overlay_bgr = cv2.addWeighted(
            original_bgr,
            0.48,
            heatmap_bgr,
            0.52,
            0
        )

        # ====================================================
        # LABEL
        # ====================================================

        cv2.rectangle(
            overlay_bgr,
            (0, 0),
            (width, 34),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            overlay_bgr,
            "Heatmap Visualization",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # SAVE
        # ====================================================

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            overlay_bgr
        )

        if not success:

            print(
                "❌ Static heatmap could not be saved"
            )

            return None

        print(
            f"✅ Static heatmap saved: "
            f"{output_path}"
        )

        return str(
            output_path
        )

    except Exception as e:

        print(
            f"❌ Static heatmap error: "
            f"{type(e).__name__}: {e}"
        )

        import traceback

        traceback.print_exc()

        return None


# ============================================================
# LOGIN
# ============================================================

def login_view(request):

    if request.user.is_authenticated:

        return redirect(
            'dashboard'
        )

    if request.method == 'POST':

        username = request.POST.get(
            'username'
        )

        password = request.POST.get(
            'password'
        )

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(
                request,
                user
            )

            messages.success(
                request,
                f'Welcome back, {username}!'
            )

            return redirect(
                'dashboard'
            )

        else:

            messages.error(
                request,
                'Invalid username or password.'
            )

    return render(
        request,
        'login.html'
    )


# ============================================================
# REGISTER
# ============================================================

def register_view(request):

    if request.user.is_authenticated:

        return redirect(
            'dashboard'
        )

    if request.method == 'POST':

        username = request.POST.get(
            'username'
        )

        password = request.POST.get(
            'password'
        )

        confirm_password = request.POST.get(
            'confirm_password'
        )

        if password != confirm_password:

            messages.error(
                request,
                'Passwords do not match.'
            )

            return render(
                request,
                'register.html'
            )

        from django.contrib.auth.models import User

        if User.objects.filter(
            username=username
        ).exists():

            messages.error(
                request,
                'Username already exists.'
            )

            return render(
                request,
                'register.html'
            )

        user = User.objects.create_user(
            username=username,
            password=password
        )

        user.save()

        messages.success(
            request,
            'Registration successful! Please login.'
        )

        return redirect(
            'login'
        )

    return render(
        request,
        'register.html'
    )


# ============================================================
# LOGOUT
# ============================================================

@login_required
def logout_view(request):

    logout(
        request
    )

    messages.info(
        request,
        'You have been logged out.'
    )

    return redirect(
        'login'
    )


# ============================================================
# DASHBOARD
# ============================================================

@login_required
def dashboard_view(request):

    media_files = (
        MediaUpload.objects
        .filter(
            user=request.user
        )
        .order_by(
            '-uploaded_at'
        )
    )

    return render(
        request,
        'dashboard.html',
        {
            'media_files':
                media_files
        }
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@login_required
def upload_page(request):

    return render(
        request,
        'upload.html'
    )


# ============================================================
# UPLOAD MEDIA
# ============================================================
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_media(request):

    print("\n")
    print("=" * 80)
    print("🚀 UPLOAD MEDIA STARTED")
    print("=" * 80)

    # ========================================================
    # REQUEST CHECK
    # ========================================================

    if (
        request.method == 'POST'
        and request.FILES.get('media_file')
    ):

        media_file = request.FILES['media_file']

        print(
            f"📁 Uploaded file: "
            f"{media_file.name}"
        )

        # ====================================================
        # FILE TYPE
        # ====================================================

        file_name = media_file.name.lower()

        is_img = is_image_file(file_name)
        is_vid = is_video_file(file_name)

        print(f"🖼️ Image: {is_img}")
        print(f"🎥 Video: {is_vid}")

        # ====================================================
        # INVALID FILE
        # ====================================================

        if not is_img and not is_vid:

            # React API request
            if request.path.startswith('/api/'):

                return JsonResponse({
                    'success': False,
                    'error': (
                        'Unsupported file format. '
                        'Please upload an image or video.'
                    )
                }, status=400)

            # Old Django frontend
            messages.error(
                request,
                'Unsupported file format. '
                'Please upload an image or video.'
            )

            return redirect('dashboard')

        # ====================================================
        # USER HANDLING
        # ====================================================

        if request.user.is_authenticated:

            upload_user = request.user

        else:

            # React frontend has no login system.
            # Use dedicated demo user.

            upload_user, created = User.objects.get_or_create(
                username='react_demo_user',
                defaults={
                    'email': 'react_demo@navsmfs.local',
                    'is_active': True,
                }
            )

        print(
            f"👤 Upload user: "
            f"{upload_user}"
        )

        # ====================================================
        # SAVE MEDIA
        # ====================================================

        media = MediaUpload.objects.create(

            user=upload_user,

            file=media_file,

            media_type=(
                'image'
                if is_img
                else 'video'
            )
        )

        file_path = media.file.path

        print(
            f"📍 File path: "
            f"{file_path}"
        )

        # ====================================================
        # VARIABLES
        # ====================================================

        faces = []

        faces_count = 0

        prediction = None

        confidence = 0.0

        real_probability = 0.0

        fake_probability = 0.0

        heatmap_overlay_path = None

        # ====================================================
        # IMAGE PROCESSING
        # ====================================================

        if is_img:

            print("\n")
            print("=" * 80)
            print("🖼️ IMAGE PROCESSING")
            print("=" * 80)

            # =================================================
            # FACE DETECTION
            # =================================================

            if detector is not None:

                try:

                    print(
                        "👤 Detecting faces..."
                    )

                    faces = detect_faces(
                        file_path,
                        detector
                    )

                    faces_count = len(faces)

                    print(
                        f"👤 Faces detected: "
                        f"{faces_count}"
                    )

                except Exception as e:

                    print(
                        f"❌ Face detection error: "
                        f"{e}"
                    )

                    faces = []

                    faces_count = 0

            # =================================================
            # PREDICTION
            # =================================================

            try:

                result = predict_image(
                    file_path
                )

                prediction = result.get(
                    'prediction',
                    'Unknown'
                )

                confidence = result.get(
                    'confidence',
                    0.0
                )

                real_probability = result.get(
                    'real_probability',
                    0.0
                )

                fake_probability = result.get(
                    'fake_probability',
                    0.0
                )

                print(
                    f"✅ Prediction: "
                    f"{prediction}, "
                    f"Confidence: "
                    f"{confidence}"
                )

            except Exception as e:

                print(
                    f"❌ Prediction error: "
                    f"{e}"
                )

                import traceback

                traceback.print_exc()

                prediction = "Error"

                confidence = 0.0

                real_probability = 0.0

                fake_probability = 0.0

            # =================================================
            # STATIC HEATMAP
            # =================================================

            print("\n")

            print(
                "🔥 Generating static heatmap..."
            )

            if (
                faces
                and
                faces_count > 0
            ):

                try:

                    # =========================================
                    # FIRST FACE
                    # =========================================

                    face = faces[0]

                    if isinstance(
                        face,
                        dict
                    ):

                        box = face.get(
                            'box',
                            [0, 0, 0, 0]
                        )

                    else:

                        box = face

                    x, y, w, h = box

                    x = max(
                        0,
                        int(x)
                    )

                    y = max(
                        0,
                        int(y)
                    )

                    w = int(w)

                    h = int(h)

                    print(
                        f"📍 Face box: "
                        f"x={x}, "
                        f"y={y}, "
                        f"w={w}, "
                        f"h={h}"
                    )

                    # =========================================
                    # READ IMAGE
                    # =========================================

                    image = cv2.imread(
                        file_path
                    )

                    if image is None:

                        print(
                            "❌ Could not read image"
                        )

                    else:

                        img_h, img_w = (
                            image.shape[:2]
                        )

                        # =====================================
                        # SAFE COORDINATES
                        # =====================================

                        x1 = max(
                            0,
                            min(
                                x,
                                img_w - 1
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                y,
                                img_h - 1
                            )
                        )

                        x2 = min(
                            img_w,
                            x + w
                        )

                        y2 = min(
                            img_h,
                            y + h
                        )

                        # =====================================
                        # FACE CROP
                        # =====================================

                        face_crop = image[
                            y1:y2,
                            x1:x2
                        ]

                        print(
                            f"📐 Face crop: "
                            f"{face_crop.shape}"
                        )

                        # =====================================
                        # CHECK CROP
                        # =====================================

                        if (
                            face_crop is not None
                            and
                            face_crop.size > 0
                        ):

                            # =================================
                            # HEATMAP DIRECTORY
                            # =================================

                            heatmap_dir = (
                                Path(
                                    settings.MEDIA_ROOT
                                )
                                /
                                'heatmaps'
                            )

                            heatmap_dir.mkdir(
                                parents=True,
                                exist_ok=True
                            )

                            # =================================
                            # FILE NAME
                            # =================================

                            heatmap_filename = (
                                f"heatmap_"
                                f"{Path(file_path).stem}_"
                                f"{media.id}.jpg"
                            )

                            heatmap_path = (
                                heatmap_dir
                                /
                                heatmap_filename
                            )

                            # =================================
                            # GENERATE HEATMAP
                            # =================================

                            generated_path = (
                                generate_static_heatmap(
                                    face_crop,
                                    heatmap_path,
                                    prediction
                                )
                            )

                            # =================================
                            # SAVE HEATMAP
                            # =================================

                            if generated_path:

                                print(
                                    "✅ Heatmap generation "
                                    "successful!"
                                )

                                media.heatmap_file = (
                                    f"heatmaps/"
                                    f"{heatmap_filename}"
                                )

                                media.save()

                                heatmap_overlay_path = (
                                    settings.MEDIA_URL
                                    +
                                    f"heatmaps/"
                                    f"{heatmap_filename}"
                                )

                                print(
                                    f"🌐 Heatmap URL: "
                                    f"{heatmap_overlay_path}"
                                )

                                # Add heatmap URL to face data

                                if isinstance(
                                    face,
                                    dict
                                ):

                                    face[
                                        'heatmap_path'
                                    ] = (
                                        heatmap_overlay_path
                                    )

                            else:

                                print(
                                    "❌ Heatmap generation failed"
                                )

                        else:

                            print(
                                "❌ Face crop is empty"
                            )

                except Exception as e:

                    print(
                        f"❌ Heatmap error: "
                        f"{type(e).__name__}: {e}"
                    )

                    import traceback

                    traceback.print_exc()

                    heatmap_overlay_path = None

            else:

                print(
                    "⚠️ No face detected. "
                    "Heatmap skipped."
                )

            # =================================================
            # ANNOTATED IMAGE
            # =================================================

            if (
                faces
                and
                faces_count > 0
            ):

                try:

                    image = cv2.imread(
                        file_path
                    )

                    if image is not None:

                        annotated_image = (
                            draw_faces_on_image(
                                image,
                                faces
                            )
                        )

                        original_path = Path(
                            file_path
                        )

                        annotated_path = (
                            original_path.parent
                            /
                            (
                                f"{original_path.stem}"
                                f"_annotated"
                                f"{original_path.suffix}"
                            )
                        )

                        cv2.imwrite(
                            str(
                                annotated_path
                            ),
                            annotated_image
                        )

                        print(
                            f"✅ Annotated image saved: "
                            f"{annotated_path}"
                        )

                except Exception as e:

                    print(
                        f"⚠️ Annotated image error: "
                        f"{e}"
                    )

        # ====================================================
        # VIDEO PROCESSING
        # ====================================================

        elif is_vid:

            print("\n")
            print("=" * 80)
            print("🎥 VIDEO PROCESSING")
            print("=" * 80)

            if detector is not None:

                try:

                    faces_count, faces = (
                        detect_faces_video(
                            file_path,
                            detector
                        )
                    )

                    print(
                        f"🎥 Faces detected in video: "
                        f"{faces_count}"
                    )

                except Exception as e:

                    print(
                        f"❌ Video processing error: "
                        f"{e}"
                    )

                    faces = []

                    faces_count = 0

        # ====================================================
        # SAVE PREDICTION DATA
        # ====================================================

        media.prediction_result = {

            'prediction':
                prediction,

            'confidence':
                float(confidence),

            'real_probability':
                float(real_probability),

            'fake_probability':
                float(fake_probability),

            'faces_count':
                int(faces_count)
        }

        media.save()

        # ====================================================
        # FINAL LOG
        # ====================================================

        print("\n")
        print("=" * 80)
        print("✅ FINAL ANALYSIS")
        print("=" * 80)

        print(
            f"Prediction: "
            f"{prediction}"
        )

        print(
            f"Confidence: "
            f"{confidence}"
        )

        print(
            f"Real probability: "
            f"{real_probability}"
        )

        print(
            f"Fake probability: "
            f"{fake_probability}"
        )

        print(
            f"Faces detected: "
            f"{faces_count}"
        )

        print(
            f"Heatmap URL: "
            f"{heatmap_overlay_path}"
        )

        print("=" * 80)

        # ====================================================
        # 🔥 REACT API RESPONSE
        # ====================================================

        if request.path.startswith('/api/'):

            print(
                "📡 Sending JSON response to React..."
            )

            # ----------------------------------------------
            # MEDIA URL
            # ----------------------------------------------

            try:

                media_url = request.build_absolute_uri(
                    media.file.url
                )

            except Exception:

                media_url = None

            # ----------------------------------------------
            # HEATMAP URL
            # ----------------------------------------------

            if heatmap_overlay_path:

                heatmap_url = (
                    request.build_absolute_uri(
                        heatmap_overlay_path
                    )
                )

            else:

                heatmap_url = None

            response_data = {

                'success':
                    True,

                'prediction':
                    str(prediction),

                'confidence':
                    float(confidence),

                'real_probability':
                    float(real_probability),

                'fake_probability':
                    float(fake_probability),

                'faces_count':
                    int(faces_count),

                'media_url':
                    media_url,

                'heatmap_url':
                    heatmap_url,
            }

            print(
                "📡 JSON RESPONSE:"
            )

            print(
                response_data
            )

            print(
                "📡 Returning HTTP 200 to React..."
            )

            return JsonResponse(
                response_data,
                status=200
            )

        # ====================================================
        # OLD DJANGO FRONTEND
        # ====================================================

        return render(
            request,
            'result.html',
            {

                'media':
                    media,

                'faces':
                    faces,

                'faces_count':
                    faces_count,

                'prediction':
                    prediction,

                'confidence':
                    confidence,

                'real_probability':
                    real_probability,

                'fake_probability':
                    fake_probability,

                'is_image':
                    is_img,

                'is_video':
                    is_vid,

                'heatmap_overlay_path':
                    heatmap_overlay_path
            }
        )

    # ========================================================
    # INVALID REQUEST
    # ========================================================

    if request.path.startswith('/api/'):

        return JsonResponse({
            'success': False,
            'error': 'No media file was uploaded.'
        }, status=400)

    return redirect(
        'dashboard'
    )


# ============================================================
# DELETE MEDIA
# ============================================================

@login_required
def delete_media(
    request,
    media_id
):

    try:

        media = MediaUpload.objects.get(
            id=media_id,
            user=request.user
        )

        # ----------------------------------------------------
        # Delete original file
        # ----------------------------------------------------

        if media.file:

            media.file.delete(
                save=False
            )

        # ----------------------------------------------------
        # Delete heatmap
        # ----------------------------------------------------

        if media.heatmap_file:

            media.heatmap_file.delete(
                save=False
            )

        # ----------------------------------------------------
        # Delete database object
        # ----------------------------------------------------

        media.delete()

        messages.success(
            request,
            'File deleted successfully.'
        )

    except MediaUpload.DoesNotExist:

        messages.error(
            request,
            'File not found.'
        )

    return redirect(
        'dashboard'
    )


# ============================================================
# MEDIA INFO API
# ============================================================

def get_media_info(
    request,
    media_id
):

    try:

        media = MediaUpload.objects.get(
            id=media_id
        )

        prediction_data = None

        # ====================================================
        # IMAGE
        # ====================================================

        if (
            media.media_type == 'image'
            and
            media.file
        ):

            file_path = media.file.path

            try:

                prediction_data = (
                    predict_image(
                        file_path
                    )
                )

            except Exception as e:

                prediction_data = {
                    "error":
                        str(e)
                }

        # ====================================================
        # RESPONSE
        # ====================================================

        return JsonResponse({

            'id':
                media.id,

            'filename':
                media.file.name,

            'media_type':
                media.media_type,

            'uploaded_at':
                media.uploaded_at.isoformat(),

            'user':
                media.user.username,

            'prediction':
                prediction_data,

            'has_heatmap':
                bool(
                    media.heatmap_file
                ),

            'heatmap_url':
                (
                    settings.MEDIA_URL
                    +
                    media.heatmap_file
                    if media.heatmap_file
                    else None
                )
        })

    except MediaUpload.DoesNotExist:

        return JsonResponse(
            {
                'error':
                    'Media not found'
            },
            status=404
        )