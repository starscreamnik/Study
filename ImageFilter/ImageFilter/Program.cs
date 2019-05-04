using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace ImageFilter
{
    class Program
    {    
        static void Main(string[] args)
        {
            var filter1 = new MatrixFilter(11);
            var image = new Bitmap(@"C:/Users/Nikita/Desktop/img.jpg");

            DateTime start = DateTime.Now;
            var res1 = filter1.RunGaussianBlur(image, false);
            TimeSpan fin = DateTime.Now - start;
            Console.WriteLine("normal mode: time = {0}", fin.ToString());
            res1.Save(@"C:/Users/Nikita/Desktop/normal.jpg");

            
            start = DateTime.Now;
            var res2 = filter1.RunGaussianBlur(image, true);
            fin = DateTime.Now - start;
            Console.WriteLine("unsafe mode: time = {0}", fin.ToString());
            res2.Save(@"C:/Users/Nikita/Desktop/.jpg");
        }
    }
}