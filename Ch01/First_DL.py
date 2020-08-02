{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "First_DL.py",
      "provenance": [],
      "authorship_tag": "ABX9TyNaDOt6bXyeMhnwtU661itn",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/surisky/DL_Class/blob/master/Ch01/First_DL.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ry11smHagZIJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sp-GUqrCoXVl",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZDH_3BTMo16-",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "np.random.seed(3)\n",
        "tf.random.set_seed(3)"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7PJ4lL20o9hI",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "Data_set=np.loadtxt(\"./Ch01/ThoraricSurgery.csv\", delimiter=\",\")"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xksysCeIpfzt",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X=Data_set[:,0:17]\n",
        "Y=Data_set[:,17]"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pdyT8XatwJAP",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model=Sequential()\n",
        "model.add(Dense(30,input_dim=17,activation='relu'))\n",
        "model.add(Dense(1,activation='sigmoid'))"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DuJmjJEwwoKZ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "b7752ad8-daa3-4e67-bdec-183d90c73a3a"
      },
      "source": [
        "model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])\n",
        "model.fit(X,Y,epochs=100,batch_size=10)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.6775 - accuracy: 0.8043\n",
            "Epoch 2/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4565 - accuracy: 0.8447\n",
            "Epoch 3/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4702 - accuracy: 0.8383\n",
            "Epoch 4/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4500 - accuracy: 0.8532\n",
            "Epoch 5/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4808 - accuracy: 0.8447\n",
            "Epoch 6/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4428 - accuracy: 0.8489\n",
            "Epoch 7/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4611 - accuracy: 0.8489\n",
            "Epoch 8/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4456 - accuracy: 0.8468\n",
            "Epoch 9/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4214 - accuracy: 0.8553\n",
            "Epoch 10/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4263 - accuracy: 0.8511\n",
            "Epoch 11/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4340 - accuracy: 0.8511\n",
            "Epoch 12/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4391 - accuracy: 0.8511\n",
            "Epoch 13/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4243 - accuracy: 0.8553\n",
            "Epoch 14/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4356 - accuracy: 0.8532\n",
            "Epoch 15/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4308 - accuracy: 0.8489\n",
            "Epoch 16/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4142 - accuracy: 0.8511\n",
            "Epoch 17/100\n",
            "47/47 [==============================] - 0s 2ms/step - loss: 0.4239 - accuracy: 0.8532\n",
            "Epoch 18/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4096 - accuracy: 0.8511\n",
            "Epoch 19/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4364 - accuracy: 0.8468\n",
            "Epoch 20/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4354 - accuracy: 0.8553\n",
            "Epoch 21/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4265 - accuracy: 0.8468\n",
            "Epoch 22/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4664 - accuracy: 0.8362\n",
            "Epoch 23/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4350 - accuracy: 0.8511\n",
            "Epoch 24/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4325 - accuracy: 0.8532\n",
            "Epoch 25/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4380 - accuracy: 0.8362\n",
            "Epoch 26/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4584 - accuracy: 0.8447\n",
            "Epoch 27/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4293 - accuracy: 0.8468\n",
            "Epoch 28/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4138 - accuracy: 0.8511\n",
            "Epoch 29/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4250 - accuracy: 0.8532\n",
            "Epoch 30/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4216 - accuracy: 0.8447\n",
            "Epoch 31/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4177 - accuracy: 0.8489\n",
            "Epoch 32/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4129 - accuracy: 0.8553\n",
            "Epoch 33/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4208 - accuracy: 0.8511\n",
            "Epoch 34/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4184 - accuracy: 0.8489\n",
            "Epoch 35/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4308 - accuracy: 0.8553\n",
            "Epoch 36/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4995 - accuracy: 0.8447\n",
            "Epoch 37/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4122 - accuracy: 0.8511\n",
            "Epoch 38/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4118 - accuracy: 0.8532\n",
            "Epoch 39/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4208 - accuracy: 0.8511\n",
            "Epoch 40/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4390 - accuracy: 0.8447\n",
            "Epoch 41/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4123 - accuracy: 0.8511\n",
            "Epoch 42/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4128 - accuracy: 0.8426\n",
            "Epoch 43/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4281 - accuracy: 0.8468\n",
            "Epoch 44/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4149 - accuracy: 0.8553\n",
            "Epoch 45/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4170 - accuracy: 0.8468\n",
            "Epoch 46/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4074 - accuracy: 0.8489\n",
            "Epoch 47/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4048 - accuracy: 0.8511\n",
            "Epoch 48/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4262 - accuracy: 0.8511\n",
            "Epoch 49/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4279 - accuracy: 0.8553\n",
            "Epoch 50/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4262 - accuracy: 0.8447\n",
            "Epoch 51/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4065 - accuracy: 0.8511\n",
            "Epoch 52/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3988 - accuracy: 0.8532\n",
            "Epoch 53/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4007 - accuracy: 0.8532\n",
            "Epoch 54/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3999 - accuracy: 0.8489\n",
            "Epoch 55/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4003 - accuracy: 0.8511\n",
            "Epoch 56/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4078 - accuracy: 0.8489\n",
            "Epoch 57/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4242 - accuracy: 0.8553\n",
            "Epoch 58/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4031 - accuracy: 0.8511\n",
            "Epoch 59/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3952 - accuracy: 0.8511\n",
            "Epoch 60/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3999 - accuracy: 0.8532\n",
            "Epoch 61/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4279 - accuracy: 0.8532\n",
            "Epoch 62/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4031 - accuracy: 0.8511\n",
            "Epoch 63/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4080 - accuracy: 0.8447\n",
            "Epoch 64/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4329 - accuracy: 0.8553\n",
            "Epoch 65/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4113 - accuracy: 0.8426\n",
            "Epoch 66/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4055 - accuracy: 0.8553\n",
            "Epoch 67/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4137 - accuracy: 0.8426\n",
            "Epoch 68/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3901 - accuracy: 0.8553\n",
            "Epoch 69/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4502 - accuracy: 0.8532\n",
            "Epoch 70/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4059 - accuracy: 0.8511\n",
            "Epoch 71/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4088 - accuracy: 0.8489\n",
            "Epoch 72/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4052 - accuracy: 0.8489\n",
            "Epoch 73/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3964 - accuracy: 0.8574\n",
            "Epoch 74/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4066 - accuracy: 0.8511\n",
            "Epoch 75/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4143 - accuracy: 0.8489\n",
            "Epoch 76/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4080 - accuracy: 0.8489\n",
            "Epoch 77/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3980 - accuracy: 0.8553\n",
            "Epoch 78/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3884 - accuracy: 0.8553\n",
            "Epoch 79/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3860 - accuracy: 0.8574\n",
            "Epoch 80/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4090 - accuracy: 0.8532\n",
            "Epoch 81/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3907 - accuracy: 0.8511\n",
            "Epoch 82/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4072 - accuracy: 0.8447\n",
            "Epoch 83/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4084 - accuracy: 0.8468\n",
            "Epoch 84/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4073 - accuracy: 0.8426\n",
            "Epoch 85/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3846 - accuracy: 0.8511\n",
            "Epoch 86/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3882 - accuracy: 0.8638\n",
            "Epoch 87/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4225 - accuracy: 0.8426\n",
            "Epoch 88/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4107 - accuracy: 0.8426\n",
            "Epoch 89/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4040 - accuracy: 0.8489\n",
            "Epoch 90/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4196 - accuracy: 0.8383\n",
            "Epoch 91/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3985 - accuracy: 0.8511\n",
            "Epoch 92/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3833 - accuracy: 0.8511\n",
            "Epoch 93/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3962 - accuracy: 0.8574\n",
            "Epoch 94/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3912 - accuracy: 0.8532\n",
            "Epoch 95/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3865 - accuracy: 0.8489\n",
            "Epoch 96/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4042 - accuracy: 0.8511\n",
            "Epoch 97/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4033 - accuracy: 0.8511\n",
            "Epoch 98/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3946 - accuracy: 0.8468\n",
            "Epoch 99/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.3852 - accuracy: 0.8489\n",
            "Epoch 100/100\n",
            "47/47 [==============================] - 0s 1ms/step - loss: 0.4026 - accuracy: 0.8553\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tensorflow.python.keras.callbacks.History at 0x7fb996087080>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xrjwyc89xCGo",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}