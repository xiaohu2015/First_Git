class Tracer(object):
    ' 装饰器：用于追踪函数调用'
    def __init__(self, func):
        self.calls = 0 #记录调用次数
        self.func = func  #将函数引用给类属性
        self.__name__ = func.__name__  #将函数保存在属性中
    def __call__(self, *args, **kwargs):
        self.calls += 1 #调用次数加1
        print('call %s to %s' % (self.calls, self.func.__name__)) #调用前输出xinxi
        result = self.func(*args, **kwargs)
        print('end!')
        return result
@Tracer                #等价于spam = Tracer(spam)
def spam(a, b, c):     #从而将spam包装到装饰器对象中
    return a + b + c
print(spam(1, 2, 3))      #实际上调用the Tracer wrapper object，激发__call__
print(spam('a', 'b', 'c'))
print(spam.__name__)


