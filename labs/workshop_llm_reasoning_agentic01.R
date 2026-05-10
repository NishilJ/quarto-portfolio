# EPPS 6323 Workshop: LLMs, Reasoning, and Agentic AI in Practice
# Companion R Script
# EPPS 6323 Knowledge Mining
# Karl Ho, University of Texas at Dallas



# 0. Setup


library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(tidyverse)
library(httr2)
library(jsonlite)

# Set API keys (replace with your own)
Sys.setenv(OPENAI_API_KEY = "")
Sys.setenv(ANTHROPIC_API_KEY = "")



# PART I: PROMPT ENGINEERING



# Q1-Q2. Prepare texts and design prompts


corp <- data_corpus_inaugural

texts_df <- tibble(
  doc_id = docnames(corp),
  text = as.character(corp),
  year = docvars(corp, "Year"),
  president = docvars(corp, "President")
) %>%
  filter(year >= 1999)

# Strategy 1: Zero-shot
prompt_zeroshot <- function(text) {
  paste0(
    "Classify the tone of this presidential inaugural address excerpt ",
    "as one of: optimistic, cautious, confrontational, unifying.\n\n",
    "Text: ", substr(text, 1, 500), "\n\n",
    "Classification:"
  )
}

# Strategy 2: Chain-of-Thought
prompt_cot <- function(text) {
  paste0(
    "Classify the tone of this presidential inaugural address excerpt ",
    "as one of: optimistic, cautious, confrontational, unifying.\n\n",
    "Think step by step:\n",
    "1. What key phrases or themes stand out?\n",
    "2. What emotional register does the speaker use?\n",
    "3. Who is the intended audience and how are they addressed?\n",
    "4. Based on your analysis, what is the overall tone?\n\n",
    "Text: ", substr(text, 1, 500), "\n\n",
    "Analysis:"
  )
}

# Strategy 3: Structured Output
prompt_structured <- function(text) {
  paste0(
    "You are an expert in political rhetoric analysis.\n\n",
    "Classify the tone of this presidential inaugural address.\n\n",
    "Respond in JSON format:\n",
    '{"tone": "optimistic|cautious|confrontational|unifying",\n',
    ' "confidence": 0.0-1.0,\n',
    ' "key_phrases": ["phrase1", "phrase2"],\n',
    ' "reasoning": "brief explanation"}\n\n',
    "Text: ", substr(text, 1, 500), "\n\n",
    "JSON:"
  )
}



# PART II: API ACCESS



# Q3. OpenAI API


call_openai <- function(prompt,
                        model = "gpt-4o-mini",
                        temperature = 0.3,
                        max_tokens = 500) {
  response <- request("https://api.openai.com/v1/chat/completions") %>%
    req_headers(
      "Authorization" = paste("Bearer", Sys.getenv("OPENAI_API_KEY")),
      "Content-Type" = "application/json"
    ) %>%
    req_body_json(list(
      model = model,
      messages = list(
        list(role = "system", content = "You are a helpful assistant."),
        list(role = "user", content = prompt)
      ),
      temperature = temperature,
      max_tokens = max_tokens
    )) %>%
    req_perform()

  resp_body_json(response)$choices[[1]]$message$content
}


# Q4. Anthropic API


call_anthropic <- function(prompt,
                           model = "claude-sonnet-4-20250514",
                           temperature = 0.3,
                           max_tokens = 500) {
  response <- request("https://api.anthropic.com/v1/messages") %>%
    req_headers(
      "x-api-key" = Sys.getenv("ANTHROPIC_API_KEY"),
      "anthropic-version" = "2023-06-01",
      "Content-Type" = "application/json"
    ) %>%
    req_body_json(list(
      model = model,
      max_tokens = max_tokens,
      messages = list(
        list(role = "user", content = prompt)
      ),
      temperature = temperature
    )) %>%
    req_perform()

  resp_body_json(response)$content[[1]]$text
}



# Q5. Batch processing


classify_texts <- function(texts_df, prompt_fn, api_fn = call_openai) {
  texts_df %>%
    mutate(
      prompt = map_chr(text, prompt_fn),
      response = map_chr(prompt, function(p) {
        Sys.sleep(1)
        tryCatch(
          api_fn(p),
          error = function(e) paste("ERROR:", e$message)
        )
      })
    )
}

# Run with API's:
results_zeroshot <- classify_texts(texts_df, prompt_zeroshot)
results_cot <- classify_texts(texts_df, prompt_cot)
results_structured <- classify_texts(texts_df, prompt_structured)



# PART III: COMPARING REASONING STRATEGIES



# Q6. Parse structured responses


parse_structured <- function(response_text) {
  clean <- str_remove_all(response_text, "```json|```")
  tryCatch(
    fromJSON(clean),
    error = function(e) list(tone = NA, confidence = NA,
                             key_phrases = NA, reasoning = NA)
  )
}


# Q7. Visualization


# Uncomment after running batch classification:
comparison_df <- bind_rows(
   results_zeroshot %>%
     mutate(strategy = "Zero-shot",
            tone = str_extract(response, "optimistic|cautious|confrontational|unifying")),
   results_cot %>%
     mutate(strategy = "Chain-of-Thought",
            tone = str_extract(response, "optimistic|cautious|confrontational|unifying")),
   results_structured %>%
     mutate(strategy = "Structured")
 ) %>%
   select(president, year, strategy, tone)
   ggplot(comparison_df, aes(x = factor(year), y = strategy, fill = tone)) +
   geom_tile(color = "white", linewidth = 1) +
   scale_fill_manual(values = c(
     "optimistic" = "#27ae60", "cautious" = "#f39c12",
    "confrontational" = "#c0392b", "unifying" = "#3498db"
   )) +
   labs(title = "Tone Classification by Prompting Strategy",
        x = "Year", y = "Strategy", fill = "Tone") +
   theme_minimal(base_size = 18) +
   theme(axis.text.x = element_text(angle = 45, hjust = 1))



# PART IV: REACT AGENT



# Q8-Q9. Tools and agent loop


tools <- list(
  search_corpus = function(query) {
    toks <- tokens(corp, remove_punct = TRUE)
    kwic_result <- kwic(toks, pattern = phrase(query), window = 10)
    if (nrow(kwic_result) == 0) return("No results found.")
    head(kwic_result, 5) %>%
      mutate(context = paste(pre, "**", keyword, "**", post)) %>%
      pull(context) %>%
      paste(collapse = "\n")
  },

  compute_stats = function(president_name) {
    sub_corp <- corpus_subset(corp, President == president_name)
    if (ndoc(sub_corp) == 0) return("President not found.")
    dfm_sub <- dfm(tokens(sub_corp, remove_punct = TRUE))
    stats <- textstat_lexdiv(dfm_sub, measure = "TTR")
    paste0("Lexical diversity (TTR) for ", president_name, ": ",
           round(stats$TTR, 3), " across ", ndoc(sub_corp), " addresses.")
  },

  kwic_search = function(pattern) {
    toks <- tokens(corp, remove_punct = TRUE)
    result <- kwic(toks, pattern = pattern, window = 8)
    if (nrow(result) == 0) return("Pattern not found.")
    head(result, 8) %>%
      mutate(line = paste0("[", docname, "] ...", pre, " **",
                           keyword, "** ", post, "...")) %>%
      pull(line) %>%
      paste(collapse = "\n")
  }
)


# Q10. Agent loop


react_agent <- function(question, tools, max_steps = 5) {
  system_prompt <- paste0(
    "You are a research assistant analyzing US inaugural addresses.\n",
    "You have access to these tools:\n",
    "- search_corpus(query): search for passages containing a phrase\n",
    "- compute_stats(president_name): get lexical diversity stats\n",
    "- kwic_search(pattern): keyword-in-context search\n\n",
    "For each step, respond in this exact format:\n",
    "Thought: [your reasoning about what to do next]\n",
    "Action: [tool_name(argument)]\n\n",
    "When you have enough information, respond:\n",
    "Thought: I have enough information.\n",
    "Answer: [your final answer]\n"
  )

  messages <- list(
    list(role = "system", content = system_prompt),
    list(role = "user", content = question)
  )

  cat("Question:", question, "\n\n")

  for (step in 1:max_steps) {
    response <- request("https://api.openai.com/v1/chat/completions") %>%
      req_headers(
        "Authorization" = paste("Bearer", Sys.getenv("OPENAI_API_KEY")),
        "Content-Type" = "application/json"
      ) %>%
      req_body_json(list(
        model = "gpt-4o-mini",
        messages = messages,
        temperature = 0.2,
        max_tokens = 300
      )) %>%
      req_perform() %>%
      resp_body_json()

    llm_text <- response$choices[[1]]$message$content
    cat(paste0("--- Step ", step, " ---\n", llm_text, "\n\n"))

    if (grepl("Answer:", llm_text)) {
      return(str_extract(llm_text, "(?<=Answer: ).*"))
    }

    action_match <- str_match(llm_text, "Action: (\\w+)\\((.+?)\\)")
    if (!is.na(action_match[1])) {
      tool_name <- action_match[2]
      tool_arg <- str_remove_all(action_match[3], "[\"']")

      if (tool_name %in% names(tools)) {
        observation <- tools[[tool_name]](tool_arg)
        cat("Observation:", substr(observation, 1, 300), "\n\n")

        messages <- c(messages, list(
          list(role = "assistant", content = llm_text),
          list(role = "user",
               content = paste("Observation:", observation))
        ))
      }
    }
  }
  "Agent reached maximum steps without a final answer."
}


# Q11 (optonal). Run agent examples


# Uncomment to run:
result1 <- react_agent(
   "How has the use of the word 'freedom' changed across inaugural addresses?",
   tools
 )
#
# result2 <- react_agent(
#   "Compare the lexical diversity of Obama and Trump's inaugural addresses.",
#   tools
# )
#
# result3 <- react_agent(
#   "What themes about the economy appear in post-2000 inaugural addresses?",
#   tools
# )
