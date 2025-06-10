import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import User
from schemas import UserType, UserInput
from strawberry.federation import Schema

Base.metadata.create_all(bind=engine)


@strawberry.type
class Query:
    @strawberry.field
    def users(self) -> list[UserType]:
        db: Session = SessionLocal()
        users = db.query(User).all()
        return [UserType(id=u.id, name=u.name, email=u.email) for u in users]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, input: UserInput) -> UserType:
        db: Session = SessionLocal()
        user = User(name=input.name, email=input.email)
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserType(id=user.id, name=user.name, email=user.email)

    @strawberry.mutation
    def update_user(self, id: int, input: UserInput) -> UserType:
        db: Session = SessionLocal()
        user = db.query(User).filter(User.id == id).first()
        user.name = input.name
        user.email = input.email
        db.commit()
        return UserType(id=user.id, name=user.name, email=user.email)

    @strawberry.mutation
    def delete_user(self, id: int) -> bool:
        db: Session = SessionLocal()
        user = db.query(User).filter(User.id == id).first()
        db.delete(user)
        db.commit()
        return True


schema = Schema(query=Query, mutation=Mutation)

app = FastAPI()
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
