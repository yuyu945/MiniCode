class SearchService:
    def query(self, text: str) -> list[str]:
        return [text]


class UserService:
    def search(self, text: str) -> list[str]:
        service = SearchService()
        return service.query(text)
