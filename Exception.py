class TrianerNotLoaded(Exception):
    def __str__(self):
        return 'trainer is not loaded'