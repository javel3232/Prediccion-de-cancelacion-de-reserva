# Usa la imagen oficial de Python como imagen base
FROM python:3.10.9

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el contenido del directorio actual al contenedor en /app
COPY . /app

# Install any required dependencies
RUN pip install -r requirements.txt
RUN pip install uvicorn


EXPOSE 8000

CMD ["uvicorn", "apirest:app", "--host", "0.0.0.0", "--port", "8000"]

