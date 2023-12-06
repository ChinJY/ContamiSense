import credentials
import openai
from Convert_to_natural_language import training_data, test_data

openai.api_key = credentials.gpt_api_key

# def chat_func(data, sample):
#     response = openai.ChatCompletion.create(
#         temperature=0,
#         model="gpt-3.5-turbo-16k-0613",
#         messages=[
#             {"role": "user", "content": "This is the data: <{}>".format(data)},
#             {"role": "system", "content": 'Based on the above data, guess if the value of "contaminated" is 1 or 0 in the following sample, shorten your response to either "1" or "0":".'},
#             {"role": "user", "content": "This is the sample: <{}>".format(sample)},
#         ]
#     )
#     return(response['choices'][0]['message']['content'])

# # print(test_data[0])

# for element in test_data:
#     print(chat_func(training_data, element))


def chat_func(sample, conversation):
    conversation.append({"role": "user", "content": "This is the sample: <{}>".format(sample)})
    response = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-3.5-turbo-1106",
        messages=conversation
    )
    return(response['choices'][0]['message']['content'])

conversation = [
    {"role": "user", "content": "This is the data: <{}>".format(training_data)},
    {"role": "system", "content": 'Based on the above data, guess if the value of "contaminated" is 1 or 0 in the following sample, shorten your response to either "1" or "0":".'},
]

for element in test_data:
    print(chat_func(element, conversation))