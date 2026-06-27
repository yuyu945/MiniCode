class UserService:
    def normalize_email(self, value: str) -> str:
        return value.strip().lower()


def normalize_email(value: str) -> str:
    service = UserService()
    return service.normalize_email(value)
