import hmac
import streamlit as st
import params

import streamlit as st
st.set_page_config(page_title=params.PAGE_TITLE,
                   page_icon=params.PAGE_ICON,
                   layout='wide')


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Adjust the ratio as needed for better centering
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        # Show input for password.
        st.text_input(
            "Password | パスワード | 密码", type="password", on_change=password_entered, key="password"
        )

        if "password_correct" in st.session_state:
            st.error("Password incorrect. | パスワードが間違っています. | 密码不对。")

    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.
