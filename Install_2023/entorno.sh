# mkdir ~/streamlit_app && cd ~/streamlit_app
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install streamlit

streamlit --version

streamlit run app.py