from __future__ import annotations

from typing import Any, Optional

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 5


def parse_response(response: requests.Response) -> tuple[bool, Any]:
    """Return a standard success flag and response payload/message."""
    if 200 <= response.status_code < 300:
        if response.status_code == 204:
            return True, "Success"
        return True, response.json()

    try:
        error_data = response.json()
        message = error_data.get("detail", "Request failed")
    except ValueError:
        message = f"Request failed with status code {response.status_code}"
    return False, message


def get_tasks(completed: Optional[bool] = None) -> tuple[bool, Any]:
    params = {}
    if completed is not None:
        params["completed"] = completed
    try:
        response = requests.get(
            f"{API_BASE_URL}/tasks",
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return parse_response(response)
    except requests.RequestException:
        return False, "Could not connect to FastAPI server."


def get_task_by_id(task_id: int) -> tuple[bool, Any]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/tasks/{task_id}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return parse_response(response)
    except requests.RequestException:
        return False, "Could not connect to FastAPI server."


def create_task(title: str, description: Optional[str], completed: bool) -> tuple[bool, Any]:
    payload = {
        "title": title,
        "description": description if description else None,
        "completed": completed,
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/tasks",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return parse_response(response)
    except requests.RequestException:
        return False, "Could not connect to FastAPI server."


def update_task(task_id: int, payload: dict[str, Any]) -> tuple[bool, Any]:
    try:
        response = requests.put(
            f"{API_BASE_URL}/tasks/{task_id}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return parse_response(response)
    except requests.RequestException:
        return False, "Could not connect to FastAPI server."


def delete_task(task_id: int) -> tuple[bool, Any]:
    try:
        response = requests.delete(
            f"{API_BASE_URL}/tasks/{task_id}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return parse_response(response)
    except requests.RequestException:
        return False, "Could not connect to FastAPI server."


st.set_page_config(page_title="Task Manager UI", page_icon="✅", layout="wide")
st.title("Task Manager Frontend")
st.caption(f"Connected API: {API_BASE_URL}")

page = st.sidebar.radio(
    "Navigation",
    ["View Tasks", "Add Task", "Get Task by ID", "Update Task", "Delete Task"],
)


if page == "View Tasks":
    st.subheader("View Tasks")
    filter_choice = st.selectbox("Filter", ["All", "Completed", "Pending"])
    refresh = st.button("Refresh Tasks")

    completed_filter: Optional[bool] = None
    if filter_choice == "Completed":
        completed_filter = True
    elif filter_choice == "Pending":
        completed_filter = False

    if refresh or True:
        ok, data = get_tasks(completed=completed_filter)
        if ok:
            if data:
                st.dataframe(data, use_container_width=True)
            else:
                st.info("No tasks found for this filter.")
        else:
            st.error(data)

elif page == "Add Task":
    st.subheader("Add Task")
    with st.form("add_task_form"):
        title = st.text_input("Title")
        description = st.text_area("Description (optional)")
        completed = st.checkbox("Completed", value=False)
        submit = st.form_submit_button("Create Task")

        if submit:
            if not title.strip():
                st.error("Title is required.")
            else:
                ok, data = create_task(title=title.strip(), description=description.strip(), completed=completed)
                if ok:
                    st.success("Task created successfully.")
                    st.json(data)
                else:
                    st.error(data)

elif page == "Get Task by ID":
    st.subheader("Get Task by ID")
    task_id = st.number_input("Task ID", min_value=1, step=1)
    if st.button("Fetch Task"):
        ok, data = get_task_by_id(int(task_id))
        if ok:
            st.success("Task found.")
            st.json(data)
        else:
            st.error(data)

elif page == "Update Task":
    st.subheader("Update Task")
    with st.form("update_task_form"):
        task_id = st.number_input("Task ID", min_value=1, step=1)
        new_title = st.text_input("New Title (optional)")
        new_description = st.text_area("New Description (optional)")
        use_completed = st.checkbox("Update completed field")
        new_completed = st.checkbox("Completed value", value=False, disabled=not use_completed)
        submit = st.form_submit_button("Update Task")

        if submit:
            update_payload: dict[str, Any] = {}
            if new_title.strip():
                update_payload["title"] = new_title.strip()
            if new_description.strip():
                update_payload["description"] = new_description.strip()
            if use_completed:
                update_payload["completed"] = new_completed

            if not update_payload:
                st.error("Provide at least one field to update.")
            else:
                ok, data = update_task(int(task_id), update_payload)
                if ok:
                    st.success("Task updated successfully.")
                    st.json(data)
                else:
                    st.error(data)

elif page == "Delete Task":
    st.subheader("Delete Task")
    task_id = st.number_input("Task ID", min_value=1, step=1)
    confirm = st.checkbox("I confirm I want to delete this task.")

    if st.button("Delete Task"):
        if not confirm:
            st.warning("Please confirm deletion first.")
        else:
            ok, data = delete_task(int(task_id))
            if ok:
                st.success("Task deleted successfully.")
            else:
                st.error(data)
