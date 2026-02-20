import streamlit as st
st.title("ESC Winding Designer")

mode = st.selectbox("Mode", ["full","adaptive","bflow"])

if st.button("Run"):
    subprocess.run(["python","-m","cli.main","--mode",mode])
