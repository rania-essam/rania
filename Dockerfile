# استخدام صورة Python الأساسية
FROM python:3.9-slim

# تعيين دليل العمل
WORKDIR /app

# تثبيت المكتبات المطلوبة للتعامل مع OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مستخدم غير root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# إنشاء وضبط صلاحيات المجلدات
RUN mkdir -p /app/uploads && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# نسخ ملفات المتطلبات
COPY --chown=appuser:appuser requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 \
    flask \
    opencv-python \
    mediapipe \
    pandas \
    numpy \
    moviepy \
    librosa \
    joblib \
    gunicorn

# تثبيت TensorFlow بشكل منفصل
RUN pip install --no-cache-dir --default-timeout=300 tensorflow

RUN mkdir -p /home/appuser/.config/matplotlib \
    && chown -R appuser:appuser /home/appuser/.config \
    && chmod -R 755 /home/appuser/.config
# نسخ الملفات المطلوبة
COPY --chown=appuser:appuser lstm_without_masking.h5 audio_model.h5 scaler.joblib ./

# نسخ باقي الملفات
COPY --chown=appuser:appuser . .

# تعيين متغيرات البيئة
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# إضافة متغير البيئة لـ Matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib

# التبديل إلى المستخدم غير root
USER appuser

# تعريف المنفذ
EXPOSE 7860

# تشغيل التطبيق
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "4", "--timeout", "300", "app:app"]