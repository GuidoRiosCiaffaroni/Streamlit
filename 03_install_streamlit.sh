sudo apt install -y python3 python3-pip python3-venv
python3 --version
pip3 --version
mkdir ~/streamlit_app && cd ~/streamlit_app
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install streamlit
streamlit --version