using System;
using System.Collections.Generic;
using System.Text;

namespace BitContainer
{
    public class BitContainer
    {
        private List<byte> data = new List<byte>();

        public int Length = 0;
        
        public bool this [int index] {
            get
            {
                var i = index / 8;
                var shift = index % 8;
                return (data[i] & (1 << shift )) != 0;
            }

            set
            {
                var i = index / 8;
                var shift = index % 8;
                if (value)
                    data[i] |= (byte) (1 << shift);
                 else
                    data[i] &= (byte) ~(1 << shift);
            }
        }

        public void PushBit(int bit)
        {
            if (Length % 8 == 0) data.Add(0);    // push fictive value to list
            this[Length] = bit != 0;
            Length++;
        }

        public void PushBit(bool bit)
        {
            if (Length % 8 == 0) data.Add(0);    // push fictive value to list
            this[Length] = bit;
            Length++;
        }

        public void Clear()
        {
            data.Clear();
            Length = 0;
        }

        public void Insert(int place, int bit)
        {
            try
            {
                if (place < 0 || place >= Length)
                    throw new IndexOutOfRangeException();
                
                if (Length % 8 == 0) data.Add(0);

                for (var i = Length; i > place; i--)
                    this[i] = this[i - 1];
                
                this[place] = bit != 0;
                Length++;
            }
            catch (IndexOutOfRangeException msg)
            {
                Console.WriteLine("INSERT: {0}", msg);
            }
        }
        
        public void Insert(int place, bool bit)
        {
            try
            {
                if (place < 0 || place >= Length)
                    throw new IndexOutOfRangeException();
                
                if (Length % 8 == 0) data.Add(0);

                for (var i = Length; i > place; i--)
                    this[i] = this[i - 1];

                this[place] = bit;
                Length++;
            }
            catch (IndexOutOfRangeException msg)
            {
                Console.WriteLine("INSERT: {0}", msg);
            }
        }

        public void Remove(int place)
        {
            try
            {
                if (place < 0 || place >= Length)
                    throw new IndexOutOfRangeException();
                
                for (var i = place; i < Length - 1; i++)
                {
                    this[i] = this[i + 1];
                }

                Length--;

                if (Length % 8 == 0)
                {
                    data.RemoveAt(data.Count - 1);
                }
            }
            catch (IndexOutOfRangeException msg)
            {
                Console.WriteLine("REMOVE: {0}", msg);
            }

        }

        public IEnumerable<bool> MyEnumerator()
        {
            for (var i = 0; i < Length; i++)
                yield return this[i];
        }

        public override string ToString()
        {
            var res = new StringBuilder();
            foreach (var bit in MyEnumerator())
                res.Append(bit?"1":"0");
            return res.ToString();
        }
    }
}