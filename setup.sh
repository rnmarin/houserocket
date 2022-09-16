mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = renan.marin@rezecon.com.br\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enabloCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
