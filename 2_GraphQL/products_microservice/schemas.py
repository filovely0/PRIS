import strawberry


@strawberry.type
class ProductType:
    id: int
    name: str
    description: str
    price: float


@strawberry.input
class ProductInput:
    name: str
    description: str
    price: float
