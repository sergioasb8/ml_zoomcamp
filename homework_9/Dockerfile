FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip install numpy tensorflow==v2.14.0 pillow
RUN pip install tensorflow

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]