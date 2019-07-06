
class ModuleConfigError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class NotBonsaiModuleError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
