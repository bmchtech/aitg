const app = new Moon({
    el: "#moon-app",
    data: {
        query: "test",
        query_num: 16,
        result_query: "",
        results: [],
        selected_result: -1,
    },
    methods: {
        submit_search: function (a, b) {
            console.log("submit_search", a, b);
            this.set("results", []);
            this.set("selected_result", -1);
            // do search
            let query = this.get("query");
            console.log("searching for: " + query);
            this.set("result_query", query);

            // this.set("results", [
            //     {
            //         title: "doc1",
            //         match: "context free grammars are <b>a type of context-free</b> language. lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
            //         score: 0.33,
            //     }
            // ]);

            // use fetch api to POST json to get results
            // fetch url encoded query
            let search_url = "/search/" + encodeURIComponent(query);
            fetch(search_url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    num_results: this.get("query_num"),
                }),
            })
                .then((response) => response.json())
                .then((resp_data) => {
                    // got results, apply them
                    console.log("got response data", resp_data);
                    let results = [];
                    resp_data.results.forEach((result) => {
                        let title = result.doc;
                        let score = result.score;
                        let sent = result.sent;
                        let context = result.context.join(" ");

                        // put sent in context
                        // find sent in context, then make it bold
                        let sent_start = context.indexOf(sent);
                        let sent_end = sent_start + sent.length;
                        let bold_context = context.substring(0, sent_start) + "<b>" + context.substring(sent_start, sent_end) + "</b>" + context.substring(sent_end);

                        results.push({
                            title: title,
                            match: bold_context,
                            score: score,
                        });
                    });
                    console.log("got results: ", results);
                    this.set("results", results);
                })
                .catch((error) => {
                    console.log("search error", error);
                });
        }
    },
});

// Moon.component("search-ui", {
//     template: `
//     `,
//     data: {
//         query: "",
//         results: [],
//     }
// });