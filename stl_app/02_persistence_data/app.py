import streamlit as st

st.title('My To-Do List Creator')

if 'my_todo_list' not in st.session_state:
    st.session_state.my_todo_list = ["Learn Markdown", "Learn Python", "Learn Streamlit"]

st.write('My current To-Do list is:', st.session_state.my_todo_list)

new_todo = st.text_input("What do you need to do?")
new_day = st.number_input(
    "How many days do you need for it?",
    min_value=0,
    step=1)

if st.button('Add the new To-Do item'):
    st.write('Adding a new item to the list')
    st.session_state.my_todo_list.append({new_todo:new_day})

st.write('My new To-Do list is:', st.session_state.my_todo_list)

# TODO
# (DONE) 1. make it save to session state
#    https://docs.streamlit.io/library/api-reference/session-state
# (DONE) 2. add a number input for number of day for each todo.
#    You may need to change the data structure of `my_todo_list`


