from django.shortcuts import render
from django.http import HttpResponse
import pickle


# Create your views here.

def home(request):
    """
    TO render home page
    :param request:
    :return:
    """
    return render(request, 'base/index.html')


def profiling(request):
    """
    To render Pandas profiling report
    :param request:
    :return:
    """
    return render(request, "base/AI4I_Profiling.html")


def predictions(request):
    try:
        processtemp = float(request.POST['processtemp'])
        rotationalspeed = float(request.POST['rotationalspeed'])
        torque = float(request.POST["torquenm"])
        toolwear = float(request.POST["toolwearmin"])
        twf = int(request.POST['toolwearf'])
        hdf = int(request.POST['heatfail'])
        pwf = int(request.POST['powerfail'])
        osf = int(request.POST['Verstrainfailure'])
        rnf = int(request.POST['randomFail'])

        # print(processtemp, rotationalspeed, torque, toolwear, twf, hdf, pwf, osf, rnf)

        # Making all the above values into StanderScaler()
        std_scaler = pickle.load(open("base//Pickle files//standard_scaler.pickle", 'rb'))

        x = [[processtemp, rotationalspeed, torque, toolwear, twf, hdf, pwf, osf, rnf]]
        x_standard = std_scaler.transform(x)

        # print("x_standard : ", x_standard)

        # Now doing prediction on scaled data
        lasso = pickle.load(open("base//Pickle files//Lasso(alpha=0.00014660150309437934).pickle", 'rb'))
        predicted_x = lasso.predict(x_standard)
        predicted_air_temp = round(predicted_x[0], 2)
        # print("predicted_air_temp : ", predicted_x)
        return render(request, "base/result.html", {'predicted_air_temp': predicted_air_temp})

    except Exception as e:
        print(f""" ERROR IN : predictions(views) : {str(e)}\n""")
