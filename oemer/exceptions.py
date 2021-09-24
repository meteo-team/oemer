# ---- Sfn exceptions ---- #
class SfnException(Exception):
    pass


class SfnNoteTrackMismatch(SfnException):
    pass


class SfnNoteGroupMismatch(SfnException):
    pass


# ---- Staffline exceptions ---- #
class StafflineException(Exception):
    pass


class StafflineCountInconsistent(StafflineException):
    pass


class StafflineNotAligned(StafflineException):
    pass


class StafflineUnitSizeInconsistent(StafflineException):
    pass
