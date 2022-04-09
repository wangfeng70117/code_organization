class _const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


const = _const()
const.MATERIAL_FLUID = 0
const.MATERIAL_SOLID = 1
const.MATERIAL_SNOW = 2

const.MARK_AIR = 0
const.MARK_FLUID = 1
const.MARK_SOLID = 2
const.MU = 0
