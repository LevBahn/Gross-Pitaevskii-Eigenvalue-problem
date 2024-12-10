# Question 1:
# Write a function to remove all duplicates from a list and return a copy of it.
# For instance, if the input list is [1, 2, 3, 3, 5, 8, 9], it should return the new list [1, 2, 3, 5, 8, 9].

def remove_duplicates(lst):
    visited_elements = set()

    for ele in lst:
        if ele not in visited_elements:
            visited_elements.add(ele)

    return visited_elements

ls1 = [1, 2, 3, 3, 5, 8, 9]
ls1_new = remove_duplicates(ls1)
print(ls1_new)

# Question 2:
# Suppose you are working in a programming language that supports three types of brackets:
# `{}`, `()`, and `[]`. Write a function to check if the brackets in a given string are balanced:
# every open bracket has a corresponding closed bracket of the correct type, and brackets may be
# nested.

# For example, “[ ( ) { (  ( ) ) ( ) } ]” is balanced, but “( ( )” and “[ ( ] )” are not.

def is_balanced(input_str: str) -> bool:

    stack = []
    for char in input_str:
        if char in '[({':
            stack.append(char)
        elif char in '])}':

            # Closing bracket without opening
            if not stack:
                return False

            # Else pop an item to check for matching
            top = stack.pop()

            if (top == '(' and char != ')') or (top == '{' and char != '}') or (top == '[' and char != ']'):
                return False

    # If an opening bracket without closing
    return len(stack) == 0

str1 = '{()}[]';
str2 = '( ( )';
str3 = '[ ( ] )';

print(is_balanced(str1))
print(is_balanced(str2))
print(is_balanced(str3))

