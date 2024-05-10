$(document).ready(function() {
    $('#videoForm').submit(function(event) {
        event.preventDefault();
        var videoId = $('#video_id').val();
        $.ajax({
            type: 'POST',
            url: '/YtAnalysis',
            contentType: 'application/json',
            data: JSON.stringify({ 'video_id': videoId }),
            success: function(response) {
                var analysisResultHtml = `
                    <div class="analysis">
                        <p>Positive Count: ${response.positive_count}</p>
                        <p>Negative Count: ${response.negative_count}</p>
                        <p>Neutral Count: ${response.neutral_count}</p>
                        <p>Uncertain Count: ${response.uncertain_count}</p>
                        <p>Litigious Count: ${response.litigious_count}</p>
                        <p><b>Overall Sentiment: ${response.overall_sentiment}</b></p>
                    </div>
                `;
                $('#analysis-result').html(analysisResultHtml);
                
                var commentsHtml = '';
                response.comments_with_sentiment.forEach(function(comment) {
                    commentsHtml += `
                        <div class="comment">
                            <p><strong>Comment:</strong> ${comment.comment}</p>
                            <p><strong>Sentiment:</strong> ${comment.sentiment}</p>
                        </div>
                    `;
                });
                $('#comments').html(commentsHtml);
            },
            error: function(error) {
                console.error('Error:', error);
            }
        });
    });
});