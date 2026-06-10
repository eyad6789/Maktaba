// Bilingual EN/AR dictionary + direction helpers.

export type Lang = 'en' | 'ar'

const dict = {
  en: {
    appName: 'Maktabah',
    appNameAlt: 'مكتبة',
    tagline: 'Ask your library anything',
    newChat: 'New chat',
    conversations: 'Conversations',
    noConversations: 'No conversations yet',
    dashboard: 'Library dashboard',
    backToChat: 'Back to chat',
    model: 'Model',
    modelAuto: 'Auto',
    modelAutoHint: 'Gemini → Claude → Local',
    unavailable: 'unavailable',
    answeredBy: 'Answered by',
    scope: 'Books',
    allBooks: 'All books',
    selectedBooks: (n: number) => `${n} book${n === 1 ? '' : 's'}`,
    searchingBooks: 'Searching the library…',
    thinking: 'Composing the answer…',
    sources: 'Sources',
    send: 'Send',
    stop: 'Stop',
    retry: 'Retry',
    useAuto: 'Switch to Auto',
    askPlaceholder: 'Ask about your books…',
    stopped: 'Generation stopped.',
    ungrounded: 'Not found in the books',
    welcomeTitle: 'The library is listening',
    welcomeSub: (books: number) =>
      books > 0
        ? `${books} book${books === 1 ? '' : 's'} ready. Ask in Arabic or English.`
        : 'Upload books from the dashboard to begin.',
    examples: [
      'What is the main idea of this book?',
      'ما الفكرة الرئيسية لهذا الكتاب؟',
      'Compare how my books define discipline',
      'لخص الفصل الأول',
    ],
    rename: 'Rename',
    delete: 'Delete',
    confirmDeleteConv: 'Delete this conversation?',
    renamePrompt: 'New title:',
    // dashboard
    library: 'Library',
    totalBooks: 'Books',
    totalChunks: 'Indexed passages',
    activeJobs: 'Active jobs',
    uploadTitle: 'Add a book',
    uploadHint: 'Drop a PDF here, or click to choose (up to 200 MB)',
    uploading: 'Uploading',
    titleOptional: 'Title (optional)',
    authorOptional: 'Author (optional)',
    duplicateBook: 'Already in the library',
    jobs: 'Vectorization',
    jobQueued: 'Queued',
    jobStarted: 'Processing',
    jobFinished: 'Completed',
    jobFailed: 'Failed',
    stageExtracting: 'Reading pages',
    stageSummarizing: 'Summarizing chapters',
    stageEmbedding: 'Embedding',
    stageUpserting: 'Indexing',
    stageDone: 'Done',
    bookTitle: 'Title',
    author: 'Author',
    language: 'Language',
    pages: 'Pages',
    chunks: 'Passages',
    status: 'Status',
    updated: 'Updated',
    confirmDeleteBook: (t: string) => `Remove “${t}” and its index?`,
    noBooks: 'No books yet — upload your first PDF above.',
    langToggle: 'عربي',
    error: 'Error',
  },
  ar: {
    appName: 'مكتبة',
    appNameAlt: 'Maktabah',
    tagline: 'اسأل مكتبتك عن أي شيء',
    newChat: 'محادثة جديدة',
    conversations: 'المحادثات',
    noConversations: 'لا توجد محادثات بعد',
    dashboard: 'لوحة المكتبة',
    backToChat: 'العودة للمحادثة',
    model: 'النموذج',
    modelAuto: 'تلقائي',
    modelAutoHint: 'Gemini ← Claude ← محلي',
    unavailable: 'غير متاح',
    answeredBy: 'أجاب',
    scope: 'الكتب',
    allBooks: 'كل الكتب',
    selectedBooks: (n: number) => `${n} ${n === 1 ? 'كتاب' : 'كتب'}`,
    searchingBooks: 'يبحث في المكتبة…',
    thinking: 'يكتب الإجابة…',
    sources: 'المصادر',
    send: 'إرسال',
    stop: 'إيقاف',
    retry: 'إعادة المحاولة',
    useAuto: 'التبديل إلى تلقائي',
    askPlaceholder: 'اسأل عن كتبك…',
    stopped: 'تم إيقاف التوليد.',
    ungrounded: 'غير موجود في الكتب',
    welcomeTitle: 'المكتبة تصغي إليك',
    welcomeSub: (books: number) =>
      books > 0
        ? `${books} ${books === 1 ? 'كتاب جاهز' : 'كتب جاهزة'}. اسأل بالعربية أو الإنجليزية.`
        : 'أضف كتبًا من لوحة المكتبة للبدء.',
    examples: [
      'ما الفكرة الرئيسية لهذا الكتاب؟',
      'What is the main idea of this book?',
      'قارن كيف تعرّف كتبي الانضباط',
      'لخص الفصل الأول',
    ],
    rename: 'إعادة تسمية',
    delete: 'حذف',
    confirmDeleteConv: 'حذف هذه المحادثة؟',
    renamePrompt: 'العنوان الجديد:',
    // dashboard
    library: 'المكتبة',
    totalBooks: 'الكتب',
    totalChunks: 'المقاطع المفهرسة',
    activeJobs: 'مهام نشطة',
    uploadTitle: 'إضافة كتاب',
    uploadHint: 'أسقط ملف PDF هنا، أو انقر للاختيار (حتى 200 م.ب.)',
    uploading: 'جارٍ الرفع',
    titleOptional: 'العنوان (اختياري)',
    authorOptional: 'المؤلف (اختياري)',
    duplicateBook: 'موجود في المكتبة من قبل',
    jobs: 'الفهرسة',
    jobQueued: 'في الانتظار',
    jobStarted: 'قيد المعالجة',
    jobFinished: 'اكتمل',
    jobFailed: 'فشل',
    stageExtracting: 'قراءة الصفحات',
    stageSummarizing: 'تلخيص الفصول',
    stageEmbedding: 'حساب المتجهات',
    stageUpserting: 'الفهرسة',
    stageDone: 'تم',
    bookTitle: 'العنوان',
    author: 'المؤلف',
    language: 'اللغة',
    pages: 'الصفحات',
    chunks: 'المقاطع',
    status: 'الحالة',
    updated: 'آخر تحديث',
    confirmDeleteBook: (t: string) => `إزالة «${t}» وفهرسه؟`,
    noBooks: 'لا توجد كتب بعد — أضف أول PDF أعلاه.',
    langToggle: 'EN',
    error: 'خطأ',
  },
} as const

export type Dict = (typeof dict)['en']

export function getDict(lang: Lang): Dict {
  return dict[lang] as Dict
}

const RTL_CHARS = /[֐-ࣿיִ-﷽ﹰ-ﻼ]/
const LTR_CHARS = /[A-Za-z]/

/** Per-content direction via the first strong character (legacy-UI parity). */
export function dirFor(text: string, fallback: Lang = 'en'): 'rtl' | 'ltr' {
  for (const ch of text) {
    if (RTL_CHARS.test(ch)) return 'rtl'
    if (LTR_CHARS.test(ch)) return 'ltr'
  }
  return fallback === 'ar' ? 'rtl' : 'ltr'
}

/** Map job state/stage identifiers to dictionary labels. */
export function jobStateLabel(d: Dict, state: string): string {
  switch (state) {
    case 'queued':
      return d.jobQueued
    case 'started':
      return d.jobStarted
    case 'finished':
      return d.jobFinished
    case 'failed':
      return d.jobFailed
    default:
      return state
  }
}

export function stageLabel(d: Dict, stage: string | null): string | null {
  switch (stage) {
    case 'extracting':
      return d.stageExtracting
    case 'summarizing':
      return d.stageSummarizing
    case 'embedding':
      return d.stageEmbedding
    case 'upserting':
      return d.stageUpserting
    case 'done':
      return d.stageDone
    default:
      return stage
  }
}
