import strawberry

@strawberry.type
class OrderType:
    id: str
    user_id: int = strawberry.field(name="userId")
    product_id: int = strawberry.field(name="productId")
    quantity: int

@strawberry.input
class OrderInput:
    user_id: int = strawberry.field(name="userId")
    product_id: int = strawberry.field(name="productId")
    quantity: int
