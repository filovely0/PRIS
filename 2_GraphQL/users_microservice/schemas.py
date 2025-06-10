import strawberry


@strawberry.type
class UserType:
    id: int
    name: str
    email: str


@strawberry.input
class UserInput:
    name: str
    email: str
