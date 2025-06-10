import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import Product
from schemas import ProductType, ProductInput
from strawberry.federation import Schema

Base.metadata.create_all(bind=engine)


@strawberry.type
class Query:
    @strawberry.field
    def products(self) -> list[ProductType]:
        db: Session = SessionLocal()
        products = db.query(Product).all()
        return [ProductType(id=p.id, name=p.name, description=p.description, price=p.price) for p in products]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_product(self, input: ProductInput) -> ProductType:
        db: Session = SessionLocal()
        product = Product(name=input.name, description=input.description, price=input.price)
        db.add(product)
        db.commit()
        db.refresh(product)
        return ProductType(id=product.id, name=product.name, description=product.description, price=product.price)

    @strawberry.mutation
    def update_product(self, id: int, input: ProductInput) -> ProductType:
        db: Session = SessionLocal()
        product = db.query(Product).filter(Product.id == id).first()
        product.name = input.name
        product.description = input.description
        product.price = input.price
        db.commit()
        return ProductType(id=product.id, name=product.name, description=product.description, price=product.price)

    @strawberry.mutation
    def delete_product(self, id: int) -> bool:
        db: Session = SessionLocal()
        product = db.query(Product).filter(Product.id == id).first()
        db.delete(product)
        db.commit()
        return True


schema = Schema(query=Query, mutation=Mutation)

app = FastAPI()
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
