from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query, status
from pydantic import BaseModel, Field


app = FastAPI(
    title="Task Manager API",
    description="A simple Task Manager API built with FastAPI wothout DB.",
    version="1.0.0",
)


class TaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=100, description="Task title")
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional task description",
    )
    completed: bool = Field(default=False, description="Task completion status")


class TaskCreate(TaskBase):
    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Learn FastAPI",
                "description": "Build a Task Manager API project",
                "completed": False,
            }
        }
    }


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    completed: Optional[bool] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Learn FastAPI deeply",
                "description": "Practice CRUD, validation, and Docker",
                "completed": True,
            }
        }
    }


class TaskResponse(TaskBase):
    id: int


tasks: dict[int, TaskResponse] = {}
next_id: int = 1


@app.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a task",
)
def create_task(task: TaskCreate) -> TaskResponse:
    global next_id

    created_task = TaskResponse(id=next_id, **task.model_dump())
    tasks[next_id] = created_task
    next_id += 1
    return created_task


@app.get(
    "/tasks",
    response_model=list[TaskResponse],
    status_code=status.HTTP_200_OK,
    summary="Get all tasks",
)
def get_tasks(
    completed: Optional[bool] = Query(
        default=None,
        description="Filter tasks by completion status",
    )
) -> list[TaskResponse]:
    all_tasks = list(tasks.values())
    if completed is None:
        return all_tasks
    return [task for task in all_tasks if task.completed == completed]


@app.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a task by ID",
)
def get_task(
    task_id: int = Path(..., ge=1, description="ID of the task to fetch")
) -> TaskResponse:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.put(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a task",
)
def update_task(
    payload: TaskUpdate,
    task_id: int = Path(..., ge=1, description="ID of the task to update"),
) -> TaskResponse:
    existing_task = tasks.get(task_id)
    if existing_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    update_data = payload.model_dump(exclude_unset=True)
    updated_task = existing_task.model_copy(update=update_data)
    tasks[task_id] = updated_task
    return updated_task


@app.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a task",
)
def delete_task(
    task_id: int = Path(..., ge=1, description="ID of the task to delete")
) -> None:
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    del tasks[task_id]
    return None
