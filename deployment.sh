sam build
sam deploy --guided
    
RewardFunctionApi=$(aws cloudformation describe-stacks --stack-name deepracer-reward \
--query 'Stacks[0].Outputs[?OutputKey==`RewardFunctionApi`].OutputValue' --output text)

sed "s~###APIEndpoint###~$RewardFunctionApi~g" reward.template > reward.py