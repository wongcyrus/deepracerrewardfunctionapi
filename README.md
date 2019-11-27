# DeepRacer Reward API


The Project details please read [AWS DeepRacer Tips and Tricks: How to build a powerful rewards function with AWS Lambda and Photoshop](https://www.linkedin.com/pulse/aws-deepracer-tips-tricks-how-build-powerful-rewards-wong/).


## Deploy with Cloud9

./resize.sh 20

wget https://gist.githubusercontent.com/wongcyrus/8eaddcc155aec4cdb451178fb5cbc2b8/raw/f60a012b9165dc9b87fd67028ba8d065be40ed2e/install_sam_cli.sh

chmod +x install_sam_cli.sh

./install_sam_cli.sh


### Open a new terminal.

git clone https://github.com/wongcyrus/deepracerrewardfunctionapi

cd deepracerrewardfunctionapi

sudo ./get_layer_packages.sh 

./deployment.sh

Now, you can copy the reward.py as the DeepRacer Reward Function.

