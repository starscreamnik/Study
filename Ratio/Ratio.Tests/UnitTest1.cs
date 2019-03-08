
namespace Ratio.Tests
{
    using System;
    using NUnit.Framework;
    [TestFixture]
    public class RatioService
    {
        private Ratio _ratioService;
        
        private static int EuclidNod(int a, int b)
        {
            while (a != 0 && b != 0)
            {
                if (a > b) a = a % b;
                else b = b % a;
            }
            return a + b;
        }

        private static int Nok(int a, int b)
        {
            return (a / EuclidNod(a, b)) * b;
        }
        [Test]
        public void SimpleRatioCreation()
        {               
            _ratioService = new Ratio(1, 1);
            Assert.IsTrue(_ratioService.Numer == _ratioService.Denom, "should be 1");
            Assert.IsTrue(_ratioService.Numer == 1, "something");
        }
        
        [TestCase(0, 5)]
        [TestCase(0, 1)]
        [TestCase(7, 3)]
        [TestCase(-1, 5)]
        public void RatioCreation(int num, int denom)
        {               
            _ratioService = new Ratio(num, denom);
            Assert.IsTrue(_ratioService.Numer * (denom/_ratioService.Denom) == num, "{0} should be stay in numerator", num);
        }
        [Test]
        public void ZeroDenominator()
        {       
            Assert.Throws<ArgumentException>(()=>_ratioService = new Ratio( 1, 0));
        }

        [Test]
        public void DivisionByZero()
        {
            var a = new Ratio(-1, 2);
            var b = new Ratio(0, 3);
            Assert.Throws<ArgumentException>(()=>_ratioService = a / b);
        }
        [Test]
        public void TypeOverflow()
        {
            var a = new Ratio(int.MaxValue & ~0x1, int.MaxValue);
            var b = new Ratio(1, 2);
            
            _ratioService = a * b;
            Console.WriteLine("{0}\t{1}", _ratioService.Numer, _ratioService.Denom);
            
            Assert.IsTrue((int.MaxValue-1)/2 == _ratioService.Numer && 
                          int.MaxValue == _ratioService.Denom,
                          "Type Overflow by multiplication");
        }
    }
}