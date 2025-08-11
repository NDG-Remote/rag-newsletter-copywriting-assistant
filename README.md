# RAG Newsletter-Copywriting Assistant

## Overview
A Retrieval Augmented Generation assistant for newsletter copywriting. It drafts on-brand subjects, preheaders, body copy and CTAs by combining your editorial guidelines, recent campaigns, and a memory that maintains tone and avoids duplicates, e.g subjects, headers, topics, and so on. Current KB uses local Markdown files, vector store is planned.

## Core Functionality
- **Copy drafting:** Generate subject lines, preheaders, body copy, CTAs.
- **Guideline checks:** Enforce style, tone, wording rules with an inline violations summary.
- **Collision detection:** Flag reused or near-duplicate subjects, angles and offers.
- **Memory:** Track decisions, banned phrases, approved claims and brand glossary.

## RAG Functionality
- **Primary context:** Local Markdown sources
- **Retriever:** Local file loader now, optional vector index later.
- **Answer strategy:** Retrieve, ground, generate, then run post-checks for guideline compliance and duplicates.

## Tools and Technologies
- **LangChain** for orchestration
- **Loaders:** Markdown file loader
- **Vector store (optional):** Chroma or FAISS
- **LLM:** OpenAI-compatible chat model, configurable via env vars
- **Duplicate check:** Local n-gram and embedding similarity
- **Prompts:** Task, style and critique prompts tailored for newsletters

## Getting Started
See **setup.md** in docs folder.
