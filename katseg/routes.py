from flask import Flask, request, render_template
from flask import current_app as app
from .inference import get_category,save_image,get_category2
from werkzeug.utils import secure_filename
import os
# app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def fragment(): 
    # Write the GET Method to get the index file
    if request.method == 'GET':
        return render_template('index.html')

    # Write the POST Method to post the results file
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File Not Uploaded')
            return
        # Read file from upload
        file = request.files['file']
        save_image(file,"input")
        # Render the result template
  
        model1 = 'modelDeepLabV3_Mila.tflite'
        # model2 = 'lite-model_deeplabv3-xception65_1_default_2.tflite'
        model3 = 'lite-model_mobilenetv2-coco_dr_1.tflite'
        model4 = 'frozen_inference_graphDLv3_mnv2_dm05_pascal.pb'
        model5= 'frozen_inference_graphDLV3_mnv2_pascal_trainaug.pb'
        get_category(img=file, model =model1 )
        # get_category(img=file, model =model2 )
        get_category(img=file, model =model3 )
        get_category2(img=file, model =model4)
        get_category2(img=file, model =model5 )
        # category2 = get_category(img=file, model ='lite-model_mobilenetv2-coco_dr_1.tflite')
        # category3 = get_category(img=file, model='lite-model_deeplabv3-xception65_1_default_2.tflite')
        # Render the result template
        return render_template('result.html', model1=model1, model3=model3,model4=model4,model5=model5)
        # from flask import Response
        # return Response( category1.getvalue(), mimetype='image/png')  # mimetype='image/png',  render_template('result.html', category=category, current_time=current_time)
        # # return Response(response = render_template('result.html'), category1.getvalue(), mimetype='image/png') # render_template('result.html', category=category, current_time=current_time)

# if __name__ == '__main__':
#     # app.run(debug=True)
#     app.run(port=33507, debug=True) #set to port 33507 so it runs in heroku