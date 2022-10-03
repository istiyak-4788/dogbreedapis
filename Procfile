web: gunicorn dogvision.wsgi
release: python manage.py makemigrations --noinput
release: python manage.py migrate --noinput
release: python manage.py migrate --run-syncdb