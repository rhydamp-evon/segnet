FROM rohitpcatalyst/abs-dj:0.1


# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy in your requirements file
COPY requirements.txt /

# This base image uses Debian Buster, whose normal package repositories have
# been archived. Point APT at the archive before installing build packages.
RUN sed -i \
    -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
    -e 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' \
    -e '/buster-updates/d' /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update

RUN set -ex \
    && BUILD_DEPS=" \
        build-essential \
        libpcre3-dev \
        libpq-dev \
        git \
        libicu-dev \
        pkg-config \
	python3-opencv \
    " \
    && apt-get install -y --no-install-recommends $BUILD_DEPS \
    # Celery 5.1.2 has legacy package metadata that pip 24.1 and newer reject.
    && pip install --upgrade pip==24.0 \
    && pip install --no-cache-dir -r /requirements.txt

RUN mkdir /atlentis-blue-sky/
WORKDIR /atlentis-blue-sky/
# COPY ./src/ /atlentis-blue-sky/
