SourceBucket=documentssettextractsource
sam package \
    --output-template-file packaged.yaml \
    --s3-bucket $SourceBucket
    
sam deploy \
    --template-file packaged.yaml \
    --stack-name deepracer-reward \
    --capabilities CAPABILITY_IAM
    
RewardFunctionApi=$(aws cloudformation describe-stacks --stack-name deepracer-reward \
--query 'Stacks[0].Outputs[?OutputKey==`RewardFunctionApi`].OutputValue' --output text)

sed "s~###APIEndpoint###~$RewardFunctionApi~g" reward.template > reward.py