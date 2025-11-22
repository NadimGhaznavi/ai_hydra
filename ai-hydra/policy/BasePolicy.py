class BasePolicy:
    # A small policy spec object. In the simple skeleton it holds:
    #  - hidden: hidden-unit count or similar hyperparam
    #  - reactive: optional reactive override (could be a function name/string or callable)
    def __init__(self, hidden: int, reactive=None):
        self.hidden = hidden
        self.reactive = reactive
