# Receipt of AI Assistants Usage
Throughout this minichallenge mainly two assistants were used: ChatGPT-4o and GitHub CoPilot.

## Use of ChatGPT
The ChatGPT assistant was mainly used for larger structural code artifacts and to help understand encountered technical error. For example, helping with structuring classes and ensuring reusability of such outsourced components.

### Prompting Strategies
- **Technical Questions**: For specific technical questions usually I tried to give GPT greater context and information so I can control the source of information in hopes of minimizing the possibility of generating hallucinations. An example would be the implementation of Recursive Feature Elimination in the context of Cross Validation: "Given the cross_validate function, should the RFE object be embedded inside the KFold CV split or outside of the split with an RFECV object?"
- **Error Resolving**: Additionally I prompted ChatGPT whenever I encountered errors I could not decypher. An example here would be "I am trying to pass an estimators Pipeline into the Dalex Explainer, however I am getting following error: [ERROR MESSAGE] What would I need to change in order for dalex to instantiate the Explainer object for just the estimator?"
- **Rewriting Help**: I sometimes found myself unsatisfied with descriptions of results. Thats where I took my took my text to ChatGPT to help me formulate the statements in a simpler way. An example prompt hereby would be "This following summary explains the results of X and Y: [SUMMARY] Help me rewrite it so the reading flow gets improved without leaving out the important findings."

## Use of GitHub CoPilot
CoPilot's purpose is mainly inline code generation. Whenever the task at hand required rather simple statements I used CoPilot. An example would be "`# Filter the two clients from client_df where account_id is either 14 or 18`". Other more complex prompts expecting more structurally complex code like "`# Function that pivots the credit and withdrawal types and aggregates their amount for all clients`" however did not work well.