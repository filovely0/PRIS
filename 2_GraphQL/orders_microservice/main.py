import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from typing import List
from strawberry.federation import Schema

from database import db
from schemas import OrderType, OrderInput
from bson.objectid import ObjectId


@strawberry.type
class Query:
    @strawberry.field
    async def orders(self) -> List[OrderType]:
        orders_cursor = db.orders.find()
        orders = await orders_cursor.to_list(length=100)
        return [
            OrderType(
                id=str(order["_id"]),
                user_id=order["user_id"],
                product_id=order["product_id"],
                quantity=order["quantity"]
            ) for order in orders
        ]


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_order(self, input: OrderInput) -> OrderType:
        order = {
            "user_id": input.user_id,
            "product_id": input.product_id,
            "quantity": input.quantity
        }
        result = await db.orders.insert_one(order)
        new_order = await db.orders.find_one({"_id": result.inserted_id})
        return OrderType(id=str(new_order["_id"]), user_id=new_order["user_id"], product_id=new_order["product_id"],
                         quantity=new_order["quantity"])

    @strawberry.mutation
    async def update_order(self, id: str, input: OrderInput) -> OrderType:
        await db.orders.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"user_id": input.user_id, "product_id": input.product_id, "quantity": input.quantity}}
        )
        updated_order = await db.orders.find_one({"_id": ObjectId(id)})
        return OrderType(id=str(updated_order["_id"]), user_id=updated_order["user_id"],
                         product_id=updated_order["product_id"], quantity=updated_order["quantity"])

    @strawberry.mutation
    async def delete_order(self, id: str) -> bool:
        result = await db.orders.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0


schema = Schema(query=Query, mutation=Mutation)

app = FastAPI()
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
