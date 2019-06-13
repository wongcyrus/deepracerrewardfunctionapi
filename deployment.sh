SourceBucket=documentssettextractsource
sam package \
    --output-template-file packaged.yaml \
    --s3-bucket $SourceBucket
    
sam deploy \
    --template-file packaged.yaml \
    --stack-name deepracer-reward \
    --capabilities CAPABILITY_IAM