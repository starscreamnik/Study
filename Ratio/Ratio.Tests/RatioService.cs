using System;
using NUnit.Framework;
namespace Ratio.Tests
{
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

        [TestCase(0, 5)]
        [TestCase(0, 1)]
        [TestCase(7, 3)]
        [TestCase(-1, 5)]
        [TestCase(-2, -7)]
        public void RatioCreation(int num, int denom)
        {
            if (EuclidNod(num, denom) != 1) return;
            
            
            _ratioService = new Ratio(num, denom);
            Assert.IsTrue(_ratioService.Numer == num, "{0} should be stay in numerator", num);
            Assert.IsTrue(_ratioService.Denom == denom, "{0} should be stay in denominator", denom);
        }

        [Test]
        public void ZeroDenominator()
        {
            try
            {
                _ratioService = new Ratio( 1, 0);
                Assert.Fail("An exception should have been thrown");
            }
            catch (ArgumentException ae)
            {
                Assert.AreEqual( "Denominator is zero", ae.Message );
            }
        }

        [Test]
        public void DivisionByZero()
        {
            var a = new Ratio(1, 2);
            var b = new Ratio(0, 3);
            var c = new Ratio(4, 0);
            try
            {
                _ratioService = a / b;
                _ratioService = a / c;
                _ratioService = c / b;
                Assert.Fail("An exception should have been thrown");
            }
            catch (ArgumentException ae)
            {
                Assert.AreEqual("Divide by zero", ae.Message);
            }
        }
    }
}