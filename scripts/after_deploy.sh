#!/bin/bash

##################
PWD='/home/ubuntu/atlentis-ml-api'
S3bucket='segnet-deployment-code-sbx'
#echo "Sync code from s3"
#sudo aws s3 sync s3://$S3bucket/ $PWD/
sudo cp -rf $PWD/.env $PWD/src/
sudo aws s3 sync s3://atlentis-blue-sky/checkpoints/segnet/production/ $PWD/src/checkpoints/
aws s3 sync s3://atlentis-blue-sky/checkpoints/fba-matting/production/ $PWD/src/checkpoints/
sudo chown -R ubuntu:ubuntu $PWD/
sudo chmod 775 -R $PWD/

########Docker compose 
cd $PWD && docker-compose build --no-cache
cd $PWD && docker-compose down 
cd $PWD && docker-compose up -d