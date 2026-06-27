class BaseImportHandler:
    def normalize_record(self, record: dict) -> dict:
        return record


class CustomerImportHandler(BaseImportHandler):
    def import_row(self, record: dict) -> dict:
        normalized = self.normalize_record(record)
        return normalized
