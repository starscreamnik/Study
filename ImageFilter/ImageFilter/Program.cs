using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace ImageFilter
{
    class Program
    {    
        static void Main(string[] args)
        {
            var filter1 = new MatrixFilter(3);
            Console.Write("Введите путь к изображению: ");
            var path = Console.ReadLine();
            var image = new Bitmap(path);

            DateTime start = DateTime.Now;
            var res1 = filter1.RunGaussianBlur(image, false);
            TimeSpan fin = DateTime.Now - start;
            Console.WriteLine("normal mode: time = {0}", fin.ToString());
            res1.Save("normal.jpg");

            
            start = DateTime.Now;
            var res2 = filter1.RunGaussianBlur(image, true);
            fin = DateTime.Now - start;
            Console.WriteLine("unsafe mode: time = {0}", fin.ToString());
            res2.Save("unsafe.jpg");
        }
    }
}