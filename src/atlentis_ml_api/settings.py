import torch
from core.networks.fba_matting import build_model
from core.services.fg_predictor import FGPredictor
import os
import environ
from pathlib import Path
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration


BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(DEBUG=(bool, False))

environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

SECRET_KEY = env("SECRET_KEY")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env("ALLOWED_HOSTS").split(" ")
APP_ENV = os.environ.get("APP_ENV", "dev")

# Application definition
INSTALLED_APPS = [
    # 'django.contrib.admin',
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "atlentis_ml_api.apps.MyAdminConfig",
    "rest_framework",
    "rest_framework_api_key",
    "training",
    "request_logs",
    "drf_spectacular",
    "chartjs",
    "bluesky",
    "object_editing",
    "adminsortable2",
    "demo",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "request_logs.middleware.RequestLogMiddleware",
]

ROOT_URLCONF = "atlentis_ml_api.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": ["templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "atlentis_ml_api.wsgi.application"

DATABASES = {
    # 'default': {
    #     'ENGINE': 'django.db.backends.sqlite3',
    #     'NAME': BASE_DIR / 'db.sqlite3',
    # }
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB"),
        "USER": env("POSTGRES_USER"),
        "PASSWORD": env("POSTGRES_PASSWORD"),
        "HOST": env("DB_HOST"),
        "PORT": env("DB_PORT"),
        "CONN_MAX_AGE": 30,
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_L10N = True

USE_TZ = True


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["atlentis_ml_api.renderers.CustomRenderer"],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework_api_key.permissions.HasAPIKey",
    ],
}

STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "static")

MEDIA_ROOT = os.path.join(BASE_DIR, "media")

USE_S3 = env("USE_S3") == "True"
if USE_S3:
    # AWS_ACCESS_KEY_ID = env('AWS_ACCESS_KEY_ID')
    # AWS_SECRET_ACCESS_KEY = env('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = env("AWS_STORAGE_BUCKET_NAME")
    AWS_S3_CUSTOM_DOMAIN = f"{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com"
    AWS_LOCATION = "assets"
    AWS_PUBLIC_MEDIA_LOCATION = "assets"
    DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
    MEDIA_URL = f"{AWS_S3_CUSTOM_DOMAIN}/{AWS_LOCATION}/"
else:
    MEDIA_URL = "/media/"

DATA_UPLOAD_MAX_MEMORY_SIZE = None  # Disable This check
FILE_UPLOAD_MAX_MEMORY_SIZE = 26214400  # 25MB

# ENV vars used for training
TRAINING_SERVER_IP = env("TRAINING_SERVER_IP")
TRAINING_SERVER_PORT = env("TRAINING_SERVER_PORT")
TRAINING_SUPERVISOR_USERNAME = env("TRAINING_SUPERVISOR_USERNAME")
TRAINING_SUPERVISOR_PASSWORD = env("TRAINING_SUPERVISOR_PASSWORD")
TRAINING_SV_START_URL = env("TRAINING_SV_START_URL")
TRAINING_SV_STOP_URL = env("TRAINING_SV_STOP_URL")

# CELERY config
CELERY_BROKER_URL = env("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND_URL = env("CELERY_RESULT_BACKEND_URL")
CELERY_ACCEPT_CONTENT = ["application/json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "Asia/Kolkata"

# Swagger config
SPECTACULAR_SETTINGS = {
    "TITLE": "Atlentis Blue Sky API",
    "DESCRIPTION": "This is the api documentation for atlentis blue sky api.",
    "VERSION": "1.0.0",
    "SERVE_PUBLIC": True,
    "SERVE_PERMISSIONS": ["rest_framework.permissions.IsAdminUser"],
    "SERVE_AUTHENTICATION": None,
    "SCHEMA_PATH_PREFIX": "/api/v[0-9]",
    "APPEND_COMPONENTS": {
        "securitySchemes": {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}
        }
    },
    "SECURITY": [
        {
            "ApiKeyAuth": [],
        }
    ]
    # OTHER SETTINGS
}

FBA_MODEL = os.path.join(BASE_DIR, "checkpoints", env("FBA_MODEL"))
SEGMENTATION_MODEL = os.path.join(BASE_DIR, "checkpoints", env("SEGMENTATION_MODEL"))

# Sentry settings
LOG_TO_SENTRY = env("LOG_TO_SENTRY") == "True"

if LOG_TO_SENTRY:
    SENTRY_DSN = env("SENTRY_DSN")
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[DjangoIntegration()],
        traces_sample_rate=0.1,
        send_default_pii=True,
    )

BLUESKY_WEBHOOK_URL = env("BLUESKY_WEBHOOK_URL")
BLUESKY_WEBHOOK_API_KEY = env("BLUESKY_WEBHOOK_API_KEY")

# Object editing inference runs in its own modern Python/CUDA service.  Keeping
# these settings optional preserves the existing Blue Sky deployment while the
# independent worker is rolled out.
OBJECT_EDITING_INFERENCE_URL = os.environ.get("OBJECT_EDITING_INFERENCE_URL", "")
OBJECT_EDITING_INFERENCE_API_KEY = os.environ.get(
    "OBJECT_EDITING_INFERENCE_API_KEY", ""
)
OBJECT_EDITING_WEBHOOK_URL = os.environ.get("OBJECT_EDITING_WEBHOOK_URL", "")
OBJECT_EDITING_WEBHOOK_API_KEY = os.environ.get(
    "OBJECT_EDITING_WEBHOOK_API_KEY", ""
)
OBJECT_EDITING_REQUEST_TIMEOUT = int(
    os.environ.get("OBJECT_EDITING_REQUEST_TIMEOUT", "300")
)


# Initializing ML model once when application starts.


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = FGPredictor(device=DEVICE, checkpoint=SEGMENTATION_MODEL)


class Args:
    encoder = "resnet50_GN_WS"
    decoder = "fba_decoder"
    weights = FBA_MODEL


args = Args()
fba_model = build_model(args)

# Remote buddy pipeline webhook url
BUDDY_WEBHOOK = env("BUDDY_WEBHOOK")

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": CELERY_BROKER_URL + "/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

SLACK_WEBHOOK = env("SLACK_WEBHOOK")
#
logsfolder = os.path.join(BASE_DIR, "logs")
if not os.path.exists(logsfolder):
    os.mkdir(logsfolder)

LOGGING = {
    "version": 1,
    "loggers": {
        "default_logger": {"handlers": ["logfile"], "level": "DEBUG"},
        "error_logger": {"handlers": ["errorlogfile"], "level": "ERROR"},
    },
    "handlers": {
        "errorlogfile": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 10,
            "backupCount": 10,
            "filename": "./logs/errors.log",
            "formatter": "verbose",
            "level": "ERROR",
        },
        "logfile": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "interval": 10,
            "backupCount": 5,
            "filename": "./logs/debug.log",
            "formatter": "verbose",
            "level": "DEBUG",
        },
    },
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        }
    },
}
