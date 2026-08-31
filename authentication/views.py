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
from django.http import JsonResponse


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ML MODEL IMPORT
# ============================================================

try:
    from ml.model_loader import predict_image, model_loader

    print("✅ ML module imported successfully!")

except ImportError as e:

    print(f"❌ Error importing ML module: {e}")

    def predict_image(image_path):
        return {
            "prediction": "AI Generated",
            "confidence": 85.0,
            "real_probability": 15.0,
            "fake_probability": 85.0,
            "error": "ML module not available"
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

    print("✅ MTCNN detector initialized successfully!")

except ImportError as e:

    print(f"❌ MTCNN not installed: {e}")

    detector = None


# ============================================================
# LOGIN
# ============================================================

def login_view(request):

    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(request, user)

            messages.success(
                request,
                f'Welcome back, {username}!'
            )

            return redirect('dashboard')

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
        return redirect('dashboard')

    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # ----------------------------------------------------
        # PASSWORD CHECK
        # ----------------------------------------------------

        if password != confirm_password:

            messages.error(
                request,
                'Passwords do not match.'
            )

            return render(
                request,
                'register.html'
            )

        # ----------------------------------------------------
        # USERNAME CHECK
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------

        user = User.objects.create_user(
            username=username,
            password=password
        )

        user.save()

        messages.success(
            request,
            'Registration successful! Please login.'
        )

        return redirect('login')

    return render(
        request,
        'register.html'
    )


# ============================================================
# LOGOUT
# ============================================================

@login_required
def logout_view(request):

    logout(request)

    messages.info(
        request,
        'You have been logged out.'
    )

    return redirect('login')


# ============================================================
# DASHBOARD
# ============================================================

@login_required
def dashboard_view(request):

    media_files = MediaUpload.objects.filter(
        user=request.user
    ).order_by('-uploaded_at')

    return render(
        request,
        'dashboard.html',
        {
            'media_files': media_files
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

@login_required
def upload_media(request):

    print("\n")
    print("=" * 80)
    print("🚨🚨🚨 UPLOAD_MEDIA FUNCTION IS RUNNING 🚨🚨🚨")
    print(f"📁 VIEWS FILE: {__file__}")
    print("=" * 80)

    # ========================================================
    # CHECK POST REQUEST
    # ========================================================

    if (
        request.method == 'POST'
        and request.FILES.get('media_file')
    ):

        media_file = request.FILES['media_file']

        print(
            f"📸 Uploaded file: "
            f"{media_file.name}"
        )

        # ====================================================
        # DETERMINE FILE TYPE
        # ====================================================

        file_name = media_file.name.lower()

        is_img = is_image_file(
            file_name
        )

        is_vid = is_video_file(
            file_name
        )

        print(
            f"🖼️ Is image: {is_img}"
        )

        print(
            f"🎥 Is video: {is_vid}"
        )

        # ====================================================
        # INVALID FORMAT
        # ====================================================

        if not is_img and not is_vid:

            messages.error(
                request,
                'Unsupported file format. '
                'Please upload an image or video.'
            )

            return redirect(
                'dashboard'
            )

        # ====================================================
        # CREATE DATABASE RECORD
        # ====================================================

        media = MediaUpload.objects.create(

            user=request.user,

            file=media_file,

            media_type=(
                'image'
                if is_img
                else 'video'
            )

        )

        file_path = media.file.path

        print(
            f"📁 Saved file path: "
            f"{file_path}"
        )

        # ====================================================
        # INITIAL VARIABLES
        # ====================================================

        faces = []

        faces_count = 0

        prediction = None

        confidence = None

        real_probability = None

        fake_probability = None

        heatmap_overlay_path = None

        # ====================================================
        # IMAGE PROCESSING
        # ====================================================

        if is_img:

            print("\n")
            print("=" * 80)
            print("🖼️ IMAGE PROCESSING STARTED")
            print("=" * 80)

            # =================================================
            # FACE DETECTION
            # =================================================

            if detector is not None:

                try:

                    print(
                        "👤 Running face detection..."
                    )

                    faces = detect_faces(
                        file_path,
                        detector
                    )

                    faces_count = len(
                        faces
                    )

                    print(
                        f"👤 Faces detected: "
                        f"{faces_count}"
                    )

                    print(
                        f"👤 Face data: "
                        f"{faces}"
                    )

                except Exception as e:

                    print(
                        f"❌ Face detection error: "
                        f"{e}"
                    )

                    faces = []

                    faces_count = 0

            else:

                print(
                    "⚠️ Detector is not initialized."
                )

                faces_count = 0

            # =================================================
            # AI PREDICTION
            # =================================================

            print("\n")
            print("=" * 80)
            print("🤖 AI PREDICTION")
            print("=" * 80)

            try:

                result = predict_image(
                    file_path
                )

                print(
                    f"📊 Raw prediction result: "
                    f"{result}"
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
                    f"{prediction}"
                )

                print(
                    f"✅ Confidence: "
                    f"{confidence}"
                )

                print(
                    f"✅ Real probability: "
                    f"{real_probability}"
                )

                print(
                    f"✅ Fake probability: "
                    f"{fake_probability}"
                )

            except Exception as e:

                print(
                    f"❌ Prediction Error: "
                    f"{type(e).__name__}: {e}"
                )

                import traceback

                traceback.print_exc()

                prediction = "Error"

                confidence = 0.0

                real_probability = 0.0

                fake_probability = 0.0

            # =================================================
            #
            # 🔴🔴🔴 EXTREME DEBUG: GRAD-CAM 🔴🔴🔴
            #
            # =================================================

            print("\n")
            print("=" * 80)
            print("🔴 GRAD-CAM DEBUG - STEP BY STEP")
            print("=" * 80)

            # -------------------------------------------------
            # STEP 1
            # -------------------------------------------------

            print(
                f"1️⃣ Faces: {faces}"
            )

            # -------------------------------------------------
            # STEP 2
            # -------------------------------------------------

            print(
                f"2️⃣ Faces count: "
                f"{faces_count}"
            )

            # =================================================
            # CHECK FACES
            # =================================================

            if (
                faces
                and
                faces_count > 0
            ):

                print(
                    "✅ Faces detected, proceeding..."
                )

                # =================================================
                # STEP 3
                # FIRST FACE
                # =================================================

                face = faces[0]

                print(
                    f"3️⃣ Face data: "
                    f"{face}"
                )

                # =================================================
                # STEP 4
                # BOUNDING BOX
                # =================================================

                x, y, w, h = face.get(
                    'box',
                    [0, 0, 0, 0]
                )

                # -------------------------------------------------
                # MTCNN sometimes returns negative x/y
                # -------------------------------------------------

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
                    f"4️⃣ Box: "
                    f"x={x}, "
                    f"y={y}, "
                    f"w={w}, "
                    f"h={h}"
                )

                # =================================================
                # VALID BOX
                # =================================================

                if (
                    w > 10
                    and
                    h > 10
                ):

                    print(
                        "✅ Valid bounding box"
                    )

                    # =================================================
                    # STEP 5
                    # READ IMAGE
                    # =================================================

                    image = cv2.imread(
                        file_path
                    )

                    print(
                        f"5️⃣ Image loaded: "
                        f"{image is not None}"
                    )

                    if image is not None:

                        # =================================================
                        # STEP 6
                        # IMAGE SHAPE
                        # =================================================

                        print(
                            f"6️⃣ Image shape: "
                            f"{image.shape}"
                        )

                        image_height, image_width = (
                            image.shape[:2]
                        )

                        # =================================================
                        # MAKE SURE BOX IS INSIDE IMAGE
                        # =================================================

                        x1 = max(
                            0,
                            min(
                                x,
                                image_width - 1
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                y,
                                image_height - 1
                            )
                        )

                        x2 = min(
                            image_width,
                            x + w
                        )

                        y2 = min(
                            image_height,
                            y + h
                        )

                        print(
                            f"📦 Safe crop coordinates: "
                            f"x1={x1}, "
                            f"y1={y1}, "
                            f"x2={x2}, "
                            f"y2={y2}"
                        )

                        # =================================================
                        # STEP 7
                        # CROP FACE
                        # =================================================

                        face_crop = image[
                            y1:y2,
                            x1:x2
                        ]

                        print(
                            f"7️⃣ Face crop shape: "
                            f"{face_crop.shape}"
                        )

                        # =================================================
                        # STEP 8
                        # CROP SIZE
                        # =================================================

                        print(
                            f"8️⃣ Face crop size: "
                            f"{face_crop.size}"
                        )

                        if face_crop.size > 0:

                            # =================================================
                            # STEP 9
                            # SAVE TEMP FACE
                            # =================================================

                            temp_face_path = str(
                                Path(file_path).with_name(
                                    f"{Path(file_path).stem}"
                                    f"_temp_face.jpg"
                                )
                            )

                            save_temp_result = (
                                cv2.imwrite(
                                    temp_face_path,
                                    face_crop
                                )
                            )

                            print(
                                f"9️⃣ Temp face saved: "
                                f"{temp_face_path}"
                            )

                            print(
                                f"   cv2.imwrite result: "
                                f"{save_temp_result}"
                            )

                            # =================================================
                            # STEP 10
                            # CHECK FILE
                            # =================================================

                            if os.path.exists(
                                temp_face_path
                            ):

                                file_size = (
                                    os.path.getsize(
                                        temp_face_path
                                    )
                                )

                                print(
                                    f"🔟 Temp file exists: "
                                    f"{file_size} bytes"
                                )

                                if file_size > 0:

                                    print(
                                        "✅ Temp file has content"
                                    )

                                    # =========================================
                                    # STEP 11
                                    # TARGET CLASS
                                    # =========================================

                                    if (
                                        prediction
                                        == 'Real'
                                    ):

                                        target_class = 1

                                    else:

                                        target_class = 0

                                    print(
                                        f"1️⃣1️⃣ Target class: "
                                        f"{target_class}"
                                    )

                                    print(
                                        "   Target meaning: "
                                        f"{'Real' if target_class == 1 else 'AI Generated'}"
                                    )

                                    # =========================================
                                    # STEP 12
                                    # CALL GRAD-CAM
                                    # =========================================

                                    print(
                                        "1️⃣2️⃣ Calling "
                                        "generate_heatmap_overlay..."
                                    )

                                    try:

                                        heatmap_result = (
                                            model_loader
                                            .generate_heatmap_overlay(
                                                temp_face_path,
                                                target_class=target_class
                                            )
                                        )

                                        # =====================================
                                        # STEP 13
                                        # PRINT RESULT
                                        # =====================================

                                        print(
                                            "1️⃣3️⃣ Heatmap result:"
                                        )

                                        print(
                                            heatmap_result
                                        )

                                        # =====================================
                                        # CHECK RESULT
                                        # =====================================

                                        if (
                                            heatmap_result
                                            and
                                            heatmap_result.get(
                                                'success',
                                                False
                                            )
                                        ):

                                            print(
                                                "✅ Heatmap generation "
                                                "SUCCESSFUL!"
                                            )

                                            # =================================
                                            # GET OVERLAY
                                            # =================================

                                            overlay = (
                                                heatmap_result.get(
                                                    'overlay'
                                                )
                                            )

                                            if (
                                                overlay
                                                is not None
                                            ):

                                                print(
                                                    "✅ Overlay exists!"
                                                )

                                                print(
                                                    f"   Overlay type: "
                                                    f"{type(overlay)}"
                                                )

                                                print(
                                                    f"   Overlay shape: "
                                                    f"{overlay.shape}"
                                                )

                                                # ==============================
                                                # HEATMAP DIRECTORY
                                                # ==============================

                                                heatmap_dir = (
                                                    Path(
                                                        settings.MEDIA_ROOT
                                                    )
                                                    / 'heatmaps'
                                                )

                                                heatmap_dir.mkdir(
                                                    parents=True,
                                                    exist_ok=True
                                                )

                                                print(
                                                    f"📁 Heatmap directory: "
                                                    f"{heatmap_dir}"
                                                )

                                                # ==============================
                                                # UNIQUE FILE NAME
                                                # ==============================

                                                heatmap_filename = (
                                                    f"heatmap_"
                                                    f"{Path(file_path).stem}_"
                                                    f"{media.id}.jpg"
                                                )

                                                heatmap_path = (
                                                    heatmap_dir
                                                    / heatmap_filename
                                                )

                                                print(
                                                    f"📄 Heatmap filename: "
                                                    f"{heatmap_filename}"
                                                )

                                                print(
                                                    f"📄 Heatmap path: "
                                                    f"{heatmap_path}"
                                                )

                                                # ==============================
                                                # RGB → BGR
                                                # ==============================

                                                overlay_bgr = (
                                                    cv2.cvtColor(
                                                        overlay,
                                                        cv2.COLOR_RGB2BGR
                                                    )
                                                )

                                                # ==============================
                                                # SAVE OVERLAY
                                                # ==============================

                                                save_heatmap = (
                                                    cv2.imwrite(
                                                        str(
                                                            heatmap_path
                                                        ),
                                                        overlay_bgr
                                                    )
                                                )

                                                print(
                                                    f"💾 Heatmap save result: "
                                                    f"{save_heatmap}"
                                                )

                                                # ==============================
                                                # VERIFY FILE
                                                # ==============================

                                                if (
                                                    save_heatmap
                                                    and
                                                    heatmap_path.exists()
                                                ):

                                                    saved_size = (
                                                        heatmap_path.stat()
                                                        .st_size
                                                    )

                                                    print(
                                                        f"✅ Heatmap physically "
                                                        f"saved!"
                                                    )

                                                    print(
                                                        f"   File size: "
                                                        f"{saved_size} bytes"
                                                    )

                                                    # ==========================
                                                    # DATABASE
                                                    # ==========================

                                                    media.heatmap_file = (
                                                        f"heatmaps/"
                                                        f"{heatmap_filename}"
                                                    )

                                                    media.save()

                                                    print(
                                                        "✅ Heatmap file "
                                                        "saved to database."
                                                    )

                                                    # ==========================
                                                    # URL
                                                    # ==========================

                                                    heatmap_overlay_path = (
                                                        media
                                                        .heatmap_file
                                                        .url
                                                    )

                                                    print(
                                                        f"🌐 Heatmap URL: "
                                                        f"{heatmap_overlay_path}"
                                                    )

                                                else:

                                                    print(
                                                        "❌ Heatmap image "
                                                        "could not be saved!"
                                                    )

                                            else:

                                                print(
                                                    "❌ Overlay is None!"
                                                )

                                        else:

                                            print(
                                                "❌ Heatmap generation "
                                                "FAILED!"
                                            )

                                            if heatmap_result:

                                                print(
                                                    "❌ Grad-CAM error:"
                                                )

                                                print(
                                                    heatmap_result.get(
                                                        'error',
                                                        'Unknown Grad-CAM error'
                                                    )
                                                )

                                            else:

                                                print(
                                                    "❌ generate_heatmap_overlay "
                                                    "returned None!"
                                                )

                                    except Exception as e:

                                        print(
                                            "❌ Exception while calling "
                                            "generate_heatmap_overlay!"
                                        )

                                        print(
                                            f"   Exception type: "
                                            f"{type(e).__name__}"
                                        )

                                        print(
                                            f"   Exception: "
                                            f"{e}"
                                        )

                                        import traceback

                                        traceback.print_exc()

                                else:

                                    print(
                                        "❌ Temp file is empty!"
                                    )

                            else:

                                print(
                                    "❌ Temp face file does not exist!"
                                )

                        else:

                            print(
                                "❌ Face crop is empty!"
                            )

                    else:

                        print(
                            "❌ Could not read uploaded image!"
                        )

                else:

                    print(
                        f"❌ Invalid bounding box: "
                        f"w={w}, h={h}"
                    )

            else:

                print(
                    "❌ No faces detected!"
                )

            # =================================================
            # GRAD-CAM DEBUG COMPLETE
            # =================================================

            print("=" * 80)

            print(
                "🔴 GRAD-CAM DEBUG COMPLETE"
            )

            print(
                f"📊 Final heatmap_overlay_path: "
                f"{heatmap_overlay_path}"
            )

            print("=" * 80)

            # =================================================
            # DRAW DETECTED FACES
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

                        image_with_faces = (
                            draw_faces_on_image(
                                image,
                                faces
                            )
                        )

                        # -----------------------------------------
                        # Safer annotated filename
                        # -----------------------------------------

                        original_path = Path(
                            file_path
                        )

                        annotated_path = (
                            original_path.parent
                            /
                            f"{original_path.stem}"
                            f"_annotated"
                            f"{original_path.suffix}"
                        )

                        cv2.imwrite(
                            str(
                                annotated_path
                            ),
                            image_with_faces
                        )

                        print(
                            f"✅ Annotated image saved: "
                            f"{annotated_path}"
                        )

                except Exception as e:

                    print(
                        f"⚠️ Error drawing faces: "
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
                        f"🎥 Video faces detected: "
                        f"{faces_count}"
                    )

                except Exception as e:

                    print(
                        f"❌ Video face detection error: "
                        f"{e}"
                    )

                    faces_count = 0

                    faces = []

            else:

                faces_count = 0

                faces = []

        # ====================================================
        # UNKNOWN / DETECTOR UNAVAILABLE
        # ====================================================

        else:

            print(
                "⚠️ Unsupported processing path."
            )

            faces_count = 0

        # ====================================================
        # SAVE PREDICTION RESULT
        # ====================================================

        media.prediction_result = {

            'prediction': prediction,

            'confidence': confidence,

            'real_probability': real_probability,

            'fake_probability': fake_probability,

            'faces_count': faces_count

        }

        # ----------------------------------------------------
        # Save media object
        # ----------------------------------------------------

        media.save()

        # ====================================================
        # FINAL DEBUG
        # ====================================================

        print("\n")
        print("=" * 80)
        print("📊 FINAL ANALYSIS RESULT")
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
            f"Faces count: "
            f"{faces_count}"
        )

        print(
            f"Heatmap URL: "
            f"{heatmap_overlay_path}"
        )

        print("=" * 80)

        # ====================================================
        # RENDER RESULT PAGE
        # ====================================================

        return render(
            request,
            'result.html',
            {

                'media': media,

                'faces': faces,

                'faces_count': faces_count,

                'prediction': prediction,

                'confidence': confidence,

                'real_probability': real_probability,

                'fake_probability': fake_probability,

                'is_image': is_img,

                'is_video': is_vid,

                'heatmap_overlay_path':
                    heatmap_overlay_path,

            }
        )

    # ========================================================
    # INVALID REQUEST
    # ========================================================

    print(
        "⚠️ Invalid upload request."
    )

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
        # Delete heatmap file
        # ----------------------------------------------------

        if media.heatmap_file:

            media.heatmap_file.delete(
                save=False
            )

        # ----------------------------------------------------
        # Delete database record
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
# GET MEDIA INFO
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
            and media.file
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
                    "error": str(e)
                }

        # ====================================================
        # JSON RESPONSE
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
                bool(media.heatmap_file)

        })

    except MediaUpload.DoesNotExist:

        return JsonResponse(
            {
                'error':
                    'Media not found'
            },
            status=404
        )