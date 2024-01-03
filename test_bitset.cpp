// bitset::to_string
#include <iostream>       // std::cout
#include <string>         // std::string
#include <bitset>         // std::bitset
#include <vector>

#include <unordered_map>
#include <functional>

typedef std::bitset<30> key;
typedef std::hash<key> hash;

int main ()
{

  std::vector<double> MI(10,0.0); // Mutual information array

  key a{27};
  key b{81};
  key c{49};
  std::unordered_map<key,int, hash> mp;
  mp.insert({a,121});
  mp.insert({b,144});
  for(auto &x: mp) {
    std::cout << x.first << "  " << x.second << '\n';
  }
  auto it = mp.find(b);
  it->second = 625;
  for(auto &x: mp) {
    std::cout << x.first << "  " << x.second << '\n';
  }
  mp[a] = 81;
  for(auto &x: mp) {
    std::cout << x.first << "  " << x.second << '\n';
  }
  mp[c] = 16;
  for(auto &x: mp) {
    std::cout << x.first << "  " << x.second << '\n';
  }
  std::cout << mp.size() << "<=\n";
  
  
  std::bitset<4> mybits;     // mybits: 0000
  mybits.set();              // mybits: 1111

  std::string mystring =
    mybits.to_string<char,std::string::traits_type,std::string::allocator_type>();

  std::cout << "mystring: " << mystring << '\n';

  int num = 30;
  int val = (num / 8) + ((num % 8) > 0);
  printf("val = %d\n",val);

  return 0;
}
