mode = st.selectbox("Mode",["full","adaptive","bflow"])

if st.button("Run"):
    subprocess.run(["python","main.py","--mode",mode])