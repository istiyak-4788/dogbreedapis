from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from app.models import Image
from app.serializers import ImageSerializer
from django.core.files.storage import FileSystemStorage
import os 
from pathlib import Path
from TF.detect import predictImage
BASE_DIR = Path(__file__).resolve().parent.parent

class ListApiView(APIView):
    def post(self, request, *args, **kwargs):  
        if request.method == "POST":
            print("request",request.data)
            print("args",args)
            serializer = ImageSerializer(data=request.data)
            if(serializer.is_valid()):
                serializer.save()
                res = predictImage.psy(serializer.data['image'])
                print(res)
                if res==1:
                    return Response({'message':"No Dog Detected"}, status=status.HTTP_404_NOT_FOUND)
                else:
                    return Response(res, status=status.HTTP_200_OK)

                